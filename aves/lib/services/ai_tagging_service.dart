import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:aves/model/app_ocr.dart';
import 'package:aves/model/app_ocrs.dart';
import 'package:aves/model/app_vector.dart';
import 'package:aves/model/entry/entry.dart';
import 'package:aves/model/entry/extensions/images.dart';
import 'package:aves/model/entry/extensions/props.dart';
import 'package:aves/services/android_debug_service.dart';
import 'package:collection/collection.dart';
import 'package:http/http.dart' as http;

/// Result of tagging one entry.
class AiTaggingOutcome {
  final int entryId;
  final bool ok;
  final Set<String> tags;
  /// The subset of [tags] that is hierarchical, e.g. `角色|Shameimaru Aya`.
  final Set<String> hierarchical;
  final Set<String> characters;
  final bool needsReview;
  /// Chinese-CLIP embedding for this photo, when the service had a model.
  final AppVector? vector;
  /// Text printed on the photo, when OCR was asked for and found any.
  final AppOcr? ocr;
  final String? error;

  const new({
    required this.entryId,
    required this.ok,
    this.tags = const {},
    this.hierarchical = const {},
    this.characters = const {},
    this.needsReview = false,
    this.vector,
    this.ocr,
    this.error,
  });

  @override
  String toString() => 'AiTaggingOutcome($entryId, ok: $ok, tags: $tags${error != null ? ', error: $error' : ''})';

  /// The tags worth storing, according to the user's preference.
  ///
  /// The tagger reports the characters separately, and the service sends each
  /// one twice: as a plain label, and as a hierarchical `角色|name` path that
  /// groups every character under one parent.  The path is what we want, so it
  /// is preferred and the plain label is only a fallback.
  Set<String> selectTags({bool includeGeneral = false}) {
    final selected = <String>{};
    for (final label in characters) {
      final path = hierarchical.firstWhereOrNull((t) => t.endsWith('|$label'));
      if (path != null) {
        selected.add(path);
      } else if (tags.contains(label)) {
        selected.add(label);
      }
    }
    if (includeGeneral) {
      // Everything that is not a character label; the character paths are
      // already in, and keeping both forms would just duplicate them.
      selected.addAll(tags.where((t) => !characters.contains(t)));
    }
    return selected;
  }
}

/// Where the time went during a run.
///
/// Exists because the slow part of this pipeline is not the tagger (about
/// 100 ms per image) but getting the thumbnail *out of the phone*: one decode,
/// one re-encode and one channel round trip per image, all of it on the device.
/// The numbers are shown in the page's log.
class AiTaggingStats {
  /// number of images a thumbnail was obtained for
  int thumbCount = 0;
  /// total time spent obtaining them, in ms
  int thumbMs = 0;
  int thumbBytes = 0;
  /// how many images were served by each method, to spot a bad fallback order
  final Map<String, int> thumbMethods = {};
  /// time spent uploading the batches, in ms
  int uploadMs = 0;
  /// time the computer reports having spent tagging, summed over batches
  int serverMs = 0;

  void addThumbnail(String method, int ms, int bytes) {
    thumbCount++;
    thumbMs += ms;
    thumbBytes += bytes;
    thumbMethods[method] = (thumbMethods[method] ?? 0) + 1;
  }

  String describe() {
    final average = thumbCount == 0 ? 0 : thumbMs ~/ thumbCount;
    final methods = thumbMethods.entries.map((e) => '${e.key}×${e.value}').join(' ');
    return '耗时：取图 $thumbCount 张 ${thumbMs}ms（均 $average ms，${methods.isEmpty ? '无' : methods}，'
        '${(thumbBytes / 1024).round()} KB），上传 ${uploadMs}ms，电脑端 ${serverMs}ms';
  }
}

/// Talks to the `albumcat` tagger service running on a computer on the same LAN.
///
/// The service only ever *returns* tag names: this class writes them to Aves'
/// own database (`AppTags`), never to the media files.
class AiTaggingService {
  static const _timeout = Duration(minutes: 5);

  /// Thrown for anything the user needs to act on (bad address, service down...).
  static String describeError(Object error) {
    final text = error is AiTaggingException ? error.message : error.toString();
    return text.replaceFirst(RegExp(r'^(Exception|SocketException|ClientException|TimeoutException): '), '');
  }

  /// `GET /health` — cheap "is the computer reachable and serving?" probe.
  static Future<Map<String, dynamic>> health(String server, {String token = ''}) async {
    final base = normalizeServer(server);
    final response = await _get('$base/health', token);
    final body = _decode(response);
    return body;
  }

  /// Where the USB tunnel puts the computer.
  ///
  /// `adb reverse tcp:8899 tcp:8899` makes the phone's *own* loopback reach the
  /// computer's server over the cable, so tagging works with no Wi-Fi at all —
  /// and the connection arrives at the server on loopback, so no firewall rule
  /// is involved either. `tools\phone_link.cmd` sets it up.
  static const int usbPort = 8899;
  static const String usbBase = 'http://127.0.0.1:$usbPort';

  /// Addresses worth trying, in order: what the user typed, then the USB tunnel.
  static List<String> candidates(String server) {
    final typed = normalizeServer(server);
    return typed == usbBase ? [typed] : [typed, usbBase];
  }

  /// Picks the first candidate whose `/health` answers, and says whether that
  /// was the USB tunnel.
  ///
  /// A reply counts even when it carries 401: the question here is reachability,
  /// and a wrong token should surface as a token error, not as "unreachable".
  /// The probe timeout stays short so a cable run does not wait out the
  /// configured address first.
  static Future<(String, bool)> resolveReachable(
    String server, {
    String token = '',
    Duration probeTimeout = const Duration(seconds: 3),
  }) async {
    Object? lastError;
    for (final base in candidates(server)) {
      try {
        await _get('$base/health', token, timeout: probeTimeout);
        return (base, base == usbBase);
      } catch (error) {
        lastError = error;
      }
    }
    throw AiTaggingException(describeError(lastError ?? '连不上电脑端。'));
  }

  /// JPEG quality the platform uses to encode the thumbnail it hands back.
  ///
  /// The debug channel's default is 100 (lossless-ish), which costs ~4x the
  /// bytes of a normal quality for no gain: the server only resizes the image
  /// down to 448 px for WD14 anyway.
  static const int thumbnailJpegQuality = 88;

  /// Tags [entries] in batches of [batchSize], with [concurrency] thumbnails
  /// fetched at a time.
  ///
  /// [extentDip] is the thumbnail size in DIP: the platform converts it to
  /// pixels, so callers usually pass `targetPixels / devicePixelRatio`.
  /// [concurrency] thumbnails are fetched at a time.
  /// Returns one outcome per entry that produced a thumbnail; entries whose
  /// thumbnail could not be obtained at all are reported with `ok: false`.
  ///
  /// Pass the base URL [resolveReachable] picked, not the raw setting, so a run
  /// over the USB tunnel is honoured.
  static Future<List<AiTaggingOutcome>> tag(
    List<AvesEntry> entries, {
    required String server,
    String token = '',
    int batchSize = 8,
    int concurrency = 4,
    double extentDip = 224,
    AiTaggingStats? stats,
    bool Function()? isCancelled,
    void Function(int done, int total)? onProgress,
    Future<void> Function(List<AiTaggingOutcome> outcomes)? onOutcomes,
    void Function(String message)? onNotice,
  }) async {
    final base = normalizeServer(server);
    final images = entries.where((e) => e.isImage).toList();
    final outcomes = <AiTaggingOutcome>[];
    if (images.isEmpty) return outcomes;

    final size = batchSize.clamp(1, 256);
    final workers = concurrency.clamp(1, 16);
    final slices = <List<AvesEntry>>[];
    for (var start = 0; start < images.length; start += size) {
      slices.add(images.sublist(start, (start + size).clamp(0, images.length)));
    }

    var done = 0;
    var failedInARow = 0;
    // Fetching a thumbnail dwarfs everything else, so the next slice is decoded
    // while the current one is being tagged: the phone and the computer end up
    // working at the same time instead of taking turns.
    var pending = _fetchSlice(slices.first, extentDip, workers, stats);
    for (var i = 0; i < slices.length; i++) {
      final fetched = await pending;
      final next = i + 1;
      if (next < slices.length) {
        pending = _fetchSlice(slices[next], extentDip, workers, stats);
      }
      if (isCancelled?.call() ?? false) break;
      final (sliceOutcomes, reached) = fetched.payload.isEmpty
          ? (<AiTaggingOutcome>[], true)
          : await _tagSlice(base, token, fetched.payload, fetched.carried, stats, onNotice);
      failedInARow = reached ? 0 : failedInARow + 1;
      if (failedInARow >= maxConsecutiveFailedBatches) {
        throw AiTaggingException('连着 $failedInARow 批都没成功，先停下来。已经打好的标签都留在手机里了。');
      }
      // Handed over slice by slice so the caller can save as it goes: an
      // interruption then costs the slices still in flight, not the whole run.
      final produced = <AiTaggingOutcome>[...fetched.failures, ...sliceOutcomes];
      outcomes.addAll(produced);
      if (onOutcomes != null) await onOutcomes(produced);
      done += slices[i].length;
      onProgress?.call(done.clamp(0, images.length), images.length);
    }
    return outcomes;
  }

  /// How many slices may fail in a row before a run gives up.
  ///
  /// One flaky slice should cost that slice rather than the run, so a slice is
  /// retried once and only a run of failures -- the computer really gone -- stops
  /// the tagging.
  static const int maxConsecutiveFailedBatches = 3;

  /// Tags one fetched slice, retrying once.
  ///
  /// Returns the outcomes and whether the computer was reached at all. A slice
  /// that stays unreachable comes back as per-entry failures, so the run carries
  /// on and reports them instead of throwing away everything before it.
  static Future<(List<AiTaggingOutcome>, bool)> _tagSlice(
    String base,
    String token,
    List<Map<String, Object?>> payload,
    Map<int, AvesEntry> carried,
    AiTaggingStats? stats,
    void Function(String message)? onNotice,
  ) async {
    Object? lastError;
    for (var attempt = 0; attempt < 2; attempt++) {
      try {
        return (await _tagBatch(base, token, payload, carried, stats), true);
      } catch (error) {
        lastError = error;
        if (attempt == 0) {
          onNotice?.call('这一批没成功，重试一次：${describeError(error)}');
        }
      }
    }
    final reason = describeError(lastError ?? '这一批没有结果');
    return (
      [
        for (final entry in carried.values)
          AiTaggingOutcome(entryId: entry.id, ok: false, error: reason),
      ],
      false,
    );
  }

  /// Base64-encodes the thumbnails of one slice, [workers] of them at a time.
  ///
  /// The platform handles `getThumbnail` on a background dispatcher
  /// (`Dispatchers.IO`), so overlapping the calls really does overlap the
  /// decodes. Results keep the slice order, so the payload stays predictable.
  static Future<
      ({
        List<Map<String, Object?>> payload,
        Map<int, AvesEntry> carried,
        List<AiTaggingOutcome> failures,
      })> _fetchSlice(List<AvesEntry> slice, double extentDip, int workers, AiTaggingStats? stats) async {
    // Fetch at tagging size, OCR included.
    //
    // An earlier version bumped this to ~1280px so that small watermarks and
    // credits would be readable. That worked, but it made every photo upload
    // 7x more bytes (24 KB -> 166 KB) to serve a case that is not what this
    // feature is for: searching the text a picture actually carries -- titles,
    // subtitles, signs, dates. Those read fine at 448px. Corner watermarks do
    // not, and that is the accepted trade: see docs/OCR.md.
    final thumbs = await _mapOrdered(slice, workers, (entry) => _thumbnailBytes(entry, extentDip, stats));
    final payload = <Map<String, Object?>>[];
    final carried = <int, AvesEntry>{};
    final failures = <AiTaggingOutcome>[];
    for (var i = 0; i < slice.length; i++) {
      final entry = slice[i];
      final thumb = thumbs[i];
      if (thumb == null || thumb.$1.isEmpty) {
        failures.add(AiTaggingOutcome(entryId: entry.id, ok: false, error: '拿不到缩略图'));
        continue;
      }
      carried[entry.id] = entry;
      payload.add({
        // the service echoes `id` back, which is how results are matched up
        'id': '${entry.id}',
        'name': entry.bestTitle,
        'size': entry.sizeBytes ?? 0,
        'image': base64Encode(thumb.$1),
        'cache': true,
        // Ask for the Chinese-CLIP vector in the same round trip: tagging then
        // doubles as indexing, so semantic search needs no second full pass.
        // A service without the model just omits the field.
        'include_vector': true,
        // Float32 so the phone can read it straight into a Float32List
        // instead of decoding half-floats by hand.
        'vector_dtype': 'float32',
        // OCR costs 2.5-7 s per photo, so it is only asked for on photos that
        // have not been read yet. The first pass over a library is slow on
        // purpose (it builds the text index); every later run is as fast as
        // before, because a row -- even an empty one meaning "no text here" --
        // is enough to skip.
        'include_ocr': appOcrs.of(entry.id) == null,
      });
    }
    return (payload: payload, carried: carried, failures: failures);
  }

  /// Runs [task] over [items] with at most [limit] in flight, keeping the order.
  ///
  /// A failing item yields null instead of aborting the whole slice: one
  /// undecodable image should not cost the tags of its neighbours.
  static Future<List<R?>> _mapOrdered<T, R>(List<T> items, int limit, Future<R?> Function(T item) task) async {
    final results = List<R?>.filled(items.length, null);
    if (items.isEmpty) return results;
    var cursor = 0;
    Future<void> worker() async {
      while (true) {
        final index = cursor++;
        if (index >= items.length) return;
        try {
          results[index] = await task(items[index]);
        } catch (_) {
          results[index] = null;
        }
      }
    }

    await Future.wait(List.generate(limit.clamp(1, items.length), (_) => worker()));
    return results;
  }

  static Future<List<AiTaggingOutcome>> _tagBatch(
    String base,
    String token,
    List<Map<String, Object?>> payload,
    Map<int, AvesEntry> carried,
    AiTaggingStats? stats,
  ) async {
    final http.Response response;
    final watch = Stopwatch()..start();
    try {
      response = await http
          .post(
            Uri.parse('$base/plan_batch'),
            headers: {
              'Content-Type': 'application/json; charset=utf-8',
              if (token.isNotEmpty) 'X-Albumcat-Token': token,
            },
            body: utf8.encode(jsonEncode(payload)),
          )
          .timeout(_timeout);
    } on TimeoutException {
      watch.stop();
      throw AiTaggingException('电脑端没有在 $base 响应（超过 ${_timeout.inMinutes} 分钟）。看看那边的窗口是不是卡住了。');
    } catch (error) {
      watch.stop();
      throw AiTaggingException(describeError(error));
    }
    watch.stop();
    if (stats != null) stats.uploadMs += watch.elapsedMilliseconds;

    final body = _decode(response);
    if (stats != null) stats.serverMs += (body['ms'] as num?)?.toInt() ?? 0;
    final results = body['results'];
    if (results is! List) {
      throw AiTaggingException('电脑端返回的内容看不懂：$body');
    }

    final outcomes = <AiTaggingOutcome>[];
    for (final raw in results) {
      if (raw is! Map) continue;
      final id = int.tryParse('${raw['id']}');
      if (id == null || !carried.containsKey(id)) continue;
      if (raw['ok'] != true) {
        outcomes.add(AiTaggingOutcome(entryId: id, ok: false, error: '${raw['error']}'));
        continue;
      }
      outcomes.add(AiTaggingOutcome(
        entryId: id,
        ok: true,
        tags: _stringSet(raw['flat']),
        hierarchical: _stringSet(raw['hierarchical']),
        characters: _stringSet(
          (raw['characters'] as List?)?.map((c) => c is Map ? '${c['label']}' : '$c'),
        ),
        needsReview: raw['needs_review'] == true,
        vector: _parseVector(raw['semantic']),
        // Presence of the key tells us OCR actually ran; a null value then
        // means "read it, found nothing", which still deserves a row.
        ocr: raw.containsKey('ocr') ? _parseOcr(raw['ocr']) : null,
      ));
    }
    return outcomes;
  }

  /// Encoded thumbnail bytes and the method that produced them, or null when
  /// nothing could be produced.
  ///
  /// `AndroidDebugService` returns already-encoded bytes (the platform encoder
  /// picks the container), which spares a decode/re-encode on our side. The
  /// channel is registered unconditionally, so it also works in release builds.
  ///
  /// Glide goes first: it downsamples to the requested extent and keeps its own
  /// cache, so it wins whenever the system thumbnail database has nothing yet
  /// (which is the normal case for a freshly scanned library). The other two
  /// are fallbacks for formats Glide cannot handle.
  ///
  /// Albumcat puts `decode` in front: doing the decode ourselves skips Glide's
  /// engine (disk cache, ARGB_8888) and the system's thumbnail database
  /// altogether, and it lets us keep RGB_565 — no alpha, so a PNG screenshot
  /// gets encoded as JPEG instead of PNG. The system methods stay as fallbacks.
  static Future<(Uint8List, String)?> _thumbnailBytes(AvesEntry entry, double extentDip, AiTaggingStats? stats) async {
    if (extentDip <= 0) extentDip = 1;
    final key = entry.getThumbnail(extent: extentDip).key;
    for (final method in const [
      ThumbnailMethod.decode,
      ThumbnailMethod.glide,
      ThumbnailMethod.mediaStore,
      ThumbnailMethod.resolver,
    ]) {
      final watch = Stopwatch()..start();
      final bytes = await AndroidDebugService.getThumbnail(key, method, jpegQuality: thumbnailJpegQuality);
      watch.stop();
      if (bytes == null || bytes.isEmpty) continue;
      // timed per method, so a failing one is paying for the next one's cache
      stats?.addThumbnail(method.name, watch.elapsedMilliseconds, bytes.length);
      return (bytes, method.name);
    }
    return null;
  }

  // -- plumbing ------------------------------------------------------------ //

  static String normalizeServer(String server) {
    var value = server.trim();
    if (value.isEmpty) throw const AiTaggingException('还没填电脑端地址。');
    if (!value.contains('://')) value = 'http://$value';
    while (value.endsWith('/')) {
      value = value.substring(0, value.length - 1);
    }
    final uri = Uri.tryParse(value);
    if (uri == null || uri.host.isEmpty) throw AiTaggingException('看不懂这个地址：「$server」');
    return value;
  }

  /// A malformed vector must never cost the tags of that photo — the two are
  /// independent results, and the vector can always be rebuilt later.
  static AppVector? _parseVector(Object? raw) {
    if (raw is! Map) return null;
    try {
      return AppVector.fromJson(Map<String, dynamic>.from(raw));
    } catch (_) {
      return null;
    }
  }

  /// Same rule as the vector: a malformed OCR block must not cost the tags.
  ///
  /// A null [raw] is not a failure -- it means the service read the photo and
  /// found no text. That is recorded as an empty [AppOcr], because "already
  /// checked, nothing there" is exactly what keeps the next run fast.
  static AppOcr? _parseOcr(Object? raw) {
    if (raw == null) {
      return AppOcr(engine: '', text: '', updated: DateTime.now().millisecondsSinceEpoch);
    }
    if (raw is! Map) return null;
    final text = raw['text'];
    if (text is! String) return null;
    return AppOcr(
      engine: '${raw['engine'] ?? ''}',
      text: text,
      updated: DateTime.now().millisecondsSinceEpoch,
    );
  }

  static Set<String> _stringSet(Object? raw) {
    // Iterable, not List: `characters` is handed in as a lazy `.map()` result,
    // which is not a List and would silently come back empty.
    if (raw is! Iterable) return {};
    return raw
        .map((v) => '$v'.trim())
        // `;` is the separator Aves uses inside a stored tag list, so a tag
        // carrying one would silently split into two
        .where((v) => v.isNotEmpty && !v.contains(';'))
        .toSet();
  }

  static Future<http.Response> _get(
    String url,
    String token, {
    Duration timeout = const Duration(seconds: 15),
  }) async {
    try {
      return await http.get(Uri.parse(url), headers: {
        if (token.isNotEmpty) 'X-Albumcat-Token': token,
      }).timeout(timeout);
    } on TimeoutException {
      throw AiTaggingException('连接超时：$url 没有响应。电脑和手机在同一个 Wi-Fi 吗？'
          '插着数据线的话，在电脑上跑一次 tools\\phone_link.cmd 就会改走 USB。');
    } catch (error) {
      throw AiTaggingException(describeError(error));
    }
  }

  static Map<String, dynamic> _decode(http.Response response) {
    if (response.statusCode == 401) {
      throw const AiTaggingException('令牌不对。电脑端用 --token 启动时，手机这里要填同一个令牌。');
    }
    if (response.statusCode != 200) {
      // The service puts a human-readable reason in `error` (e.g. "text tower
      // not loaded" on a 503). Prefer it over dumping raw JSON at the user.
      String? reason;
      try {
        final body = jsonDecode(utf8.decode(response.bodyBytes));
        if (body is Map && body['error'] != null) reason = '${body['error']}';
      } catch (_) {
        // not JSON: fall back to the truncated body below
      }
      throw AiTaggingException('电脑端返回 HTTP ${response.statusCode}：${reason ?? _short(response.body)}');
    }
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (decoded is! Map) throw AiTaggingException('电脑端返回的不是 JSON 对象：${_short(response.body)}');
    if (decoded['ok'] == false) throw AiTaggingException('电脑端报错：${decoded['error']}');
    return decoded.cast<String, dynamic>();
  }

  static String _short(String text) => text.length <= 200 ? text : '${text.substring(0, 200)}...';
}

class AiTaggingException implements Exception {
  final String message;

  const new(this.message);

  @override
  String toString() => message;
}
