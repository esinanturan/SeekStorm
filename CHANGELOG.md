# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.27] - 2025-05-14

### Fixed

- very rare position compression bug fixed.

## [0.12.26] - 2025-05-13

### Fixed

- Put winapi crate behind conditional compilation #[cfg(target_os = "windows")]

## [0.12.25] - 2025-05-12

### Improved

- Faster index_document, commit, clear_index: 
  Increased SEGMENT_KEY_CAPACITY prevents HashMap resizing during indexing.
  vector/hashmap reuse instead of reinitialization.

### Fixed

- Intersection between RLE-RLE and RLE-Bitmap compressed posting lists fixed.

## [0.12.24] - 2025-05-02

### Fixed

- Fixes a 85% performance drop (Windows 11 24H2/Intel hybrid CPUs only) caused by a faulty Windows 11 24H2 update,  
  that changed the task scheduler behavior into under-utilizing the P-Cores over E-cores of Intel hybrid CPUs.
  This is a workaround until Microsoft fixes the issue in a future update.
  The fix solves the issue for the SeekStorm server, if you embedd the SeekStorm library into your own code you have to apply the fix as well.
  See blog post for details: https://seekstorm.com/blog/80-percent-performance-drop/

## [0.12.23] - 2025-04-28

### Added

- Ingestion of files in [CSV](https://en.wikipedia.org/wiki/Comma-separated_values), SSV, TSV, PSV format with `ingest_csv()` method and seekstorm_server command line `ingest`:  
  configurable header, delimiter char, quoting, number of skipped document, number of indexed documents.
- `stop_words` parameter (predefined languages and custom) added to create_index IndexMetaObject:  
  Stop words are not indexed for compact index and faster queries.
- `frequent_words` parameter (predefined languages and custom) added to create_index IndexMetaObject:  
  consecutive frequent words are indexed as n-gram combinations for short posting lists and fast phrase queries.
- TokenizerType `Whitespace` and `WhitespaceLowercase` added.
- `truncate()` and `substring()` utils.

## [0.12.22] - 2025-04-16

### Fixed

- Problem fixed where an intersection in an very small index didn't return results (all_terms_frequent).
- Early termination fixed in single_blockid: did not guarantee most relevant results for filtered single term queries with result type Topk.

## [0.12.21] - 2025-03-27

### Added

- Stemming for 18 languages added: new property `IndexMetaObject.stemmer: StemmerType` 
- Allow to specify the markup tags to insert **before** and **after** each highlighted term. Default is "<b>" "</b>" (by @DanLLC).

### Changed

- Updated [OpenAPI documents](https://github.com/SeekStorm/SeekStorm/tree/main/src/seekstorm_server) directory.

### Fixed

- Fixed read_f32() in utils.rs

## [0.12.20] - 2025-03-05

### Changed

- PDF ingestion via `pdfium` dependency moved behind a new `pdf` feature flag which is enabled by default.  
  You can disable the SeekStorm default features by using `seekstorm = { version = "0.12.19", default-features = false }` in the cargo.toml of your application.  
  This can be useful to reduce the size of your application or if there are dependency version conflicts.
- feature flags documented in README.md

## [0.12.19] - 2025-03-04

### Fixed

- Backward compatibility to indexes created prior v0.12.18 restored.

## [0.12.18] - 2025-03-03

### Fixed

- Fixes intersection_vector16 for target_arch != "x86_64".
- Fixes issue where multiple indices per API key were not correctly reloaded after server restart (IndexMetaObject.id #[serde(skip)] removed).
- Fixes issue #39 with commandline() in server.rs for docker environment without -ti parameter (run interactively with a tty session).

### Changed

- Updated to Rust edition 2024.
- Changed serde_json::from_str(&value.to_string()).unwrap_or(value.to_string()).to_string() -> serde_json::from_value::<String>(value.clone()).unwrap_or(value.to_string())

## [0.12.17] - 2025-02-15

### Fixed

- Fixed issue in clear_index.

## [0.12.16] - 2025-02-14

### Added

- Basic tests added (issue #33): cargo test
- New method current_doc_count() returns the number of indexed documents - deleted documents.

### Fixed

- Fixed issue #32 in clear_index.

## [0.12.15] - 2025-02-12

### Fixed

- Fixed issue #34 - refactoring of http_server (by @gabriel-v)

## [0.12.14] - 2025-02-10

### Fixed

- Fixed issue #36 panic at realtime search
- Fixed a possible issue in clear_index

## [0.12.13] - 2025-02-08

### Improved

- Intersection speed for ResultType::Count improved.

### Fixed

- Fixes issue #31 for queries with query parameter length=0 and ResultType::TopK or ResultType::TopkCount. 
  - If you specify length=0, resultType::TopkCount will automatically downgraded to resultType::Count and return the number of results only, without returning the results itself.
  - If you don't specify the length in the REST API, a default of 10 will be used.

## [0.12.12] - 2025-02-07

### Fixed

- Fixes an issue in clear_index that prevented the facet.json file from being created in commit after clear_index, causing problems after reloading the index. Fixes issue #27.

## [0.12.11] - 2025-02-03

### Fixed

- clear_index fixed. Fixes issue #26 .

## [0.12.10] - 2025-02-01

### Fixed

- Fixed indexing postings with more than 8_192 positions.
- Fixed an issue with Chinese word segmentation, where a hyphen within a string was interpreted as a NOT ('-') operator in front of one of the resulting segmented words.
- Fixed an issue with NOT query terms that are RLE compressed.
- Fixed an issue for union > 8 terms with custom result sorting.

## [0.12.9] - 2025-01-29

### Fixed

- Automatic resize of postings_buffer in index_posting.
- Fixed a subtract with overflow exception when real time search was enabled.
- Fixed exception if > 10 query terms.
- Fixed stack overflow in some long union queries.
- Fixed endless loop while intersecting RLE-compressed posting lists.
- Updated rand from v0.8.5 to v0.9.0.

## [0.12.8] - 2025-01-24

### Fixed

- Removed unsafe std::slice::from_raw_parts to cast arrays of different element types, which caused unaligned data exceptions.

## [0.12.7] - 2025-01-19

### Fixed

- Endless loop while intersecting multiple RLE-compressed posting lists fixed. Fixes issue #21 .

## [0.12.6] - 2025-01-16

### Fixed

- Exception while intersecting multiple RLE-compressed posting lists fixed. Fixes issue #21 .

## [0.12.5] - 2025-01-14

### Fixed

- Endless loop while intersecting multiple RLE-compressed posting lists fixed. Fixes issue #21 .

## [0.12.4] - 2025-01-12

### Fixed

- Fixed a subtract with overflow exception that occurred in debug mode when committing, and the previous commit was < 64k documents. Fixes issue #22 .
- Changed cast_byte_ushort_slice and cast_byte_ulong_slice to either take mutable references and return mutable ones or take immutable references and return immutable ones.
- Added index_file and docstore_file flush in commit.

## [0.12.3] - 2024-12-21

### Added

- Docker file and container added (#17).
- https://hub.docker.com/r/wolfgarbe/seekstorm_server
- `docker run -ti -p "8000:80" wolfgarbe/seekstorm_server:v0.12.3`
- Added a server welcome web page with instructions how to create an API key and index.

### Fixed

- Exception handling if docker is run without the -ti parameter (run interactively with a tty session).

## [0.12.2] - 2024-12-14

### Changed

- hyper crate upgraded from v0.14.31 to v1.5.1

## [0.12.1] - 2024-12-12

### Added

- Delete Documents(s) REST API endpoint now supports deleting all documents in the index (clear index).

### Changed

- REST API documentation improved.
- Library documentation improved.
- All public methods, structs, enums and properties got document comments. 
- Stricter linting: missing_docs, unused_import_braces, trivial_casts, trivial_numeric_casts, unused_qualifications.

## [0.12.0] - 2024-12-11

### Added

- Code first OpenAPI documentation generation added for SeekStorm server REST API. 
- New console command `openapi` to create `openapi.json` and `openapi.yml`.
- Pregenerated [openapi files](https://github.com/SeekStorm/SeekStorm/tree/main/src/seekstorm_server) directory.
- SeekStorm server [REST API online documentation](https://seekstorm.apidocumentation.com/).
- Constructor for SchemaField added.

## [0.11.1] - 2024-12-05

### Changed

- In index_document skip tokenizing fields with !schema_field.indexed.
- New property SchemaField.field_id to indicate the field order in the schema map.
- In master.js, the fields in the schema map are now sorted by field_id to preserve the field order of the schema.json.
- In the Web UI, the preview panel is now always using 100% height, independent from the number of results in the result panel.

## [0.11.0] - 2024-11-28

### Added

- New tokenizer UnicodeAlphanumericZH (enable SeekStorm Cargo feature 'zh')
  - Implements Chinese word segmentation to segment continuous Chinese text into tokens for indexing and search.
  - Supports mixed Latin and Chinese texts
  - Supports Chinese sentence boundary chars for KWIC snippets ahd highlighting.
- get_synonyms, set_synonyms and add_synonyms library methods and REST API endpoints added 
  - Updated synonyms only affect subsequently indexed documents.

### Changed

- Commit now waits until all previously started index_posting have finished (multi-threading).

## [0.10.0] - 2024-11-25

### Added

- One-way and multi-way synonym definition per index added
  - Synonyms parameter added to create_index (both library and REST API).
  - Allows to define term synonyms per index.
  - Supports both one-way and multi-way synonyms.
  - Synonym support for result highlighting.
  - Currently only single term synonyms without spaces are supported.
- New server console command `create` to manually create a demo API key (`delete` to delete the demo API key and asociated indices).

## [0.9.0] - 2024-11-18

### Added

- PDF ingestion: Ingest PDF files and directories including sub-directories.  
- Ingest via console or REST API, search via Web UI or REST API.
- Stores original PDF file in index, served via REST API endpoint.
- Embedded PDF viewer in Web UI. 
- Result sorting in Web UI (Score/Date/Price/Distance acending/descending).  
- Numeric facet filter and histogram in Web UI (Date/Price/Distance...).  
- Min/Max numeric facet aggregation (Date/Price/Distance...).

- Library
  - `create_index` now creates a `files` subdirectory in index, to store copies of ingested PDF files.
  - new public method `ingest_pdf`: ingests a given pdf file or all pdf files in a given path, recursively.
  - new public methods `index_pdf_bytes` and `index_pdf_file`
    - converts pdf to text and indexes it
    - extracts title from metatag, or first line of text, or from filename
    - extracts creation date from metatag, or from file creation date (Unix timestamp: the number of seconds since 1 January 1970)
    - copy_file in index_pdf copies all ingested pdf files to "files" subdirectory in index
  - new public method `get_file`: gets a file from the "files" sub-directory in index
  - new FieldType `Timestamp`, identical to I64, but enables an UI to interpret I64 as timestamp without resorting to a specific field name as indicator.
  - new min/max value detection for numeric facet fields.
  - new public method `get_index_facets_minmax` (min/max aggregation values of numerical facet fields).

- Server
  - new REST API endpoint `index_file`: POST api/v1/index/{index_id}/file/{doc_id} 
    calls index_file_api, index_pdf_bytes and indexes the PDF file in the specified api_key and index_id
  - new REST API endpoint `get_file`:   GET  api/v1/index/{index_id}/file/{doc_id}
    calls get_file_api, get_file and returns the PDF file associated with the doc_id in the specified api_key and index_id with "Content-Type", "application/pdf"
  - REST API error messages added
  - Web UI automatically uses one of three options for result preview: 
    - result links to public web URL (e.g. Wikipedia articles) or
    - fields with path to private PDF documents via REST API (e.g. PDF ingestion) or
    - text fields.
  - WEB UI automatically displays both PDF previews (PDF ingestion) and text previews (JSON ingestion) when hovering with the mouse over the result list.
  - WEB UI now with result sort: score/newest/oldest.
  - seekstorm readme updated ([PDF search demo](https://github.com/SeekStorm/SeekStorm/?tab=readme-ov-file#build-a-pdf-search-engine-with-the-seekstorm-server)). 
  - seekstorm_server readme updated ([Console commands](https://github.com/SeekStorm/SeekStorm/blob/main/src/seekstorm_server/README.md#console-commands)).
  - If the server console command `ingest` is used with a single path parameter only, then default demo API key and index_id=0 are used.
  - New [command line parameter](https://github.com/SeekStorm/SeekStorm/tree/main/src/seekstorm_server#command-line-parameters) `ingest_path` to set the default path for data files to ingest with the console command `ingest`, if entered without absolute path/filename.
  - `get_index_stats_api` returns now `facets_minmax` property from `get_index_facets_minmax` (min/max aggregation values of numerical facet fields).
  - `SearchResultObject.facets` changed from `Vec<Facet>` to `HashMap<String,Facet>` (`HashMap<field_name,Vec<facet_label,facet_count>>`)

## [0.8.0] - 2024-10-30

### Added

- Help for server console commands added
- Added a hint of color to console messages
- Preparation for ingest of local files in PDF format (WIP).

### Changed

- TokenizerType::UnicodeAlphanumeric and TokenizerType::UnicodeAlphanumericFolded now allow '+' '-' '#' in middle or end of a term: c++, c#, block-max, top-k

### Fixed

- query_type_default for server REST API and for library fixed
- multifield decoding case fixed: three embedded fields with each a single position
- Checks if path/file exists for ingest console command

## [0.7.6] - 2024-10-25

### Changed

- Set default query_type to Intersection for REST API, if not specified in SearchRequestObject
- Set default similarity_type to Bm25fProximity for REST API, if not specified in CreateIndexRequest
- Set default tokenizer_type to UnicodeAlphanumeric for REST API, if not specified in CreateIndexRequest

## [0.7.5] - 2024-10-24

### Fixed

- Indexing of documents with > 2 fields fixed

## [0.7.4] - 2024-10-22

### Fixed

- Search with more than 2 indexed fields fixed

## [0.7.3] - 2024-10-18

### Fixed

- Fixed BM25 component precalculation for get_bm25f_multiterm_multifield

## [0.7.2] - 2024-10-18

### Fixed

- Multi-platform build fixed

## [0.7.1] - 2024-10-18

### Fixed

- Parameters of intersection_vector16 method fixed for aarch64 build

## [0.7.0] - 2024-10-11

### Added

- The SeekStorm server now allows to ingest local data files in [JSON](https://en.wikipedia.org/wiki/JSON), [Newline-delimited JSON](https://github.com/ndjson/ndjson-spec) (ndjson), and [Concatenated JSON](https://en.wikipedia.org/wiki/JSON_streaming) format via console `ingest [data_filename] [api_key] [index_id]` command.  
  The document ingestion is streamed without loading the whole document vector into memory to allow for unlimited file size while keeping RAM consumption low.
- ingest_json() is also available as public method in the SeekStorm library.
- The [Embedded web UI](https://github.com/SeekStorm/SeekStorm/blob/main/src/seekstorm_server#open-embedded-web-ui-in-browser) of SeekStorm multi-tenancy server now allows to search and display results from any index in your web browser without coding.  
  The field names to display in the web UI can be automatically detected or pre-defined. 
- seekstorm_server readme updated ([src/seekstorm_server/README.md](https://github.com/SeekStorm/SeekStorm/blob/main/src/seekstorm_server/README.md#command-line-parameters)).
- Optionally specify the default query type (Union/Intersection) via REST API (Intersection is default if not specified).

### Changed

- jQuery v3.3.1 updated to v3.7.1 (used in server web UI).

## [0.6.2] - 2024-10-10

### Fixed

- Exception fixed that occurred when intersecting terms with RLE compressed posting list [(#5)](https://github.com/SeekStorm/SeekStorm/issues/5). 

## [0.6.1] - 2024-10-05

### Fixed

- Exception fixed that occurred when intersecting more than two terms, including two terms with RLE compressed posting list [(#5)](https://github.com/SeekStorm/SeekStorm/issues/5). 
- Exception fixed that occurred when searching terms with 65.536 postings per block. The fix requires reindexing if your index has been affected!

## [0.6.0] - 2024-09-30

### Added

- Range query facets for Point field type added: Calculating the distance between a specified facet field of type Point and a base Point, in kilometers or miles,  
  using Euclidian distance (Pythagoras theorem) with Equirectangular approximation. The distance ranges are equivalent to concentric circles of different radius around the base point.
  The numbers of matching results that fall into the defined range query facet buckets are counted and returned in the ResultObject facets property.

### Changed

- Filtering of facet field type Point changed from distance to distance range, to make behaviour equivalent to the other numerical field types.

### Fixed

- Search query_facets, facets_filter, result_sort parameter documentation fixed.

## [0.5.0] - 2024-09-25

### Added

- **New Unicode character folding/normalization tokenizer** (diacritics, accents, umlauts, bold, italic, full-width ...)  
  **'TokenizerType::UnicodeAlphanumericFolded'** and method **fold_diacritics_accents_zalgo_umlaut()** 
  convert text with **diacritics, accents, zalgo text, umlaut, bold, italic, full-width UTF-8 characters** into its basic representation.  
  Unicode UTF-8 has made life so much easier compared to the old code pages, but its endless possibilities also pose challenges in parsing and indexing.  
  The challenge is that the same basic letter might be represented by different UTF8 characters if they contain diacritics, accents, or are bold, italic, or full-width.  
  Sometimes, users can't search because the keyboard doesn't have these letters or they don't know how to enter, or they even don't know what that letter looks like.  
  Sometimes the document to be ingested is already written without diacritics for the same reasons.  
  We don't want to search for every variant separately, most often we even don't know that they exist in the index.  
  We want to have all results, for every variant, no matter which variant is entered in the query, 
  e.g. for indexing LinkedIn posts that make use of external bold/italic formatters or for indexing documents in accented languages.  
  It is important that the search engine supports the character folding rather than external preprocessing, as we want to have both:  
  **enter the query in any character form**, **receive all results independent from their character form**, but **have them returned in their original, unaltered characters**.
- **New apostroph handling** in **'TokenizerType::UnicodeAlphanumericFolded'** prevents that short meaningless term parts preceding or following the apostroph get indexed (e.g. "s" in "someone's") or become part of the query.

## [0.4.0] - 2024-09-20

### Added

- **Sorting of results by any facet field** stated in the result_sort search parameter, ascending or descending.  
  Multiple sort fields are combined by "sort by, then sort by" ("tie-breaking"-algorithm).  
  After all sort fields are considered, the results are sorted by BM25 score as last clause.  
  If no sort fields are stated, then the results are just sorted by BM25 score.
  Currently, sorting by fields is more expensive, as it prevents WAND search acceleration.

- **Geo proximity search**
  - New field type Point: array of 2 * f64 coordinate values (longitude and latitude) as defined in https://geojson.org.  
    Internally encoded into a single 64 bit Morton code for efficient range queries.  
    get_facet_value decodes Morton code back to Point.  
  - New public methods: encode_morton_2_d(point:&Point)->u64, decode_morton_2_d(code: u64)->Point,  
    euclidian_distance(point1: &Point, point2: &Point, unit: &DistanceUnit)->f64,  
    point_distance_to_morton_range(point: &Point,distance: f64,unit: &DistanceUnit)-> Range<u64>  
  - New property ResultSort.base that allows to specify a query base value (e.g. of type Point for current location coordinate) during search.  
    Then results won't be ordered by the field values but by the distance between the field value and query base value, e.g. for geo proximity search.
  - New parameter distance_fields: &Vec<DistanceField> for the get_document method allows to insert distance fields into result documents,  
    calculating the distance between a specified facet field of type Point and a base Point, in kilometers or miles,  
    using Euclidian distance (Pythagoras theorem) with Equirectangular approximation.
  - SearchRequestObject in query_index_api and GetDocumentRequest in get_document_api (server REST API) have a new property distance_fields: Vec<DistanceField>.

- **get_facet_value()** : Returns value from facet field for a doc_id even if schema stored=false (field not stored in document JSON).  
  Facet fields are more compact than fields stored in document JSON.
  Facet fields are faster because no document loading, decompression and JSON decoding is required.  
  Facet fields are always memory mapped, internally always stored with fixed byte length layout.
  Facet fields are instantly updatable without re-indexing the whole document.

## [0.3.0] - 2024-08-26

### Added

- Multi-value string facets added (FieldType::StringSet)

### Improved

- get_index_string_facets refactored

### Fixed

- facet_filter field not found exception handling

## [0.2.0] - 2024-08-21

### Added

- String & Numeric Range Facets implemented: Counting and filtering of matching results.
- Supports u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, String types for facet.
- Numeric RangeType CountAboveRange/CountBelowRange/CountWithinRange implemented.
- Schema has a new field property: "field_facet":true (default=false).
- Index has a new get_index_string_facets method: returns a list of facet fields, each with field name and a list of unique values and their count (number of times a specific value appears in the whole index).
- The search ResultObject now has a facets property. Returns index facets (value appears in the whole index) if querystring.is_empty, otherwise query facets (values appear in docs that match the query).
- [FACETED_SEARCH.md](https://github.com/SeekStorm/SeekStorm/blob/main/FACETED_SEARCH.md) : Introduction to faceted search

### Improved

- BM25 component precalculation.
- Query performance improved.
- Documentation improved.
- ARCHITECTURE.md updated.

### Fixed

- Result count for unions fixed if the index contained deleted documents.

### Changed

- Schema (breaking change: properties renamed)

## [0.1.18] - 2024-07-30

### Added

- delete_document, delete_documents, delete_documents_by_query, update_document, update_documents implemented, both for library and server REST API (single docid/document and vector of docid/document)
- Query index REST API now returns also _id and _score fields for each result
- Documentation updated
- test_api.rest updated
- SearchRequestObject contains result_type parameter (Count, Topk, TopkCount)

### Fixed

- Fixed count_total for realtime search with field filters

## [0.1.17] - 2024-07-18

### Changed

- "detect longest field id" only if index_mut.indexed_field_vec.len()>1
- ResultListObject.result_count_estimated (count) removed
- ResultListObject.result_count_total i64 -> usize; serde rename countEvaluated removed
- count_evaluated renamed to count_total in master.js and api_endpoints.rs
- early termination invoked earlier in intersection_blockid
- sort_by -> sort_unstable_by
- Normal intersection with galloping when SIMD not supported for the architecture.

## [0.1.16] - 2024-07-16

### Added

- Add support for processor architectures other than x86 (i.e. Apple Silicon) (#1)

## [0.1.15] - 2024-06-14

### Improved

- Query performance improved.
- Documentation improved.
- New mute parameter for open_index and create_index: prevents emitting of status messages, e.g. when using pipes. 

### Changed

- commit changed from Index/sync to IndexArc/async

## [0.1.14] - 2024-06-12

### Fixed

- Fix corruption of a committed document that could occur under specific conditions.

## [0.1.13] - 2024-06-03

### Fixed

- Intermittent indexing with multiple commits of incomplete levels (< 65_536 documents) fixed
- get_document of uncommitted docs with a docid within the 64k range of an already committed incomplete level fixed
- Result count for searches that both include results from uncommitted and committed documents fixed
- Server handles empty index directories (manually deleted files) gracefully

### Improved

- Index compression ratio improved

## [0.1.12] - 2024-03-25

### Fixed

- Indexing (field_indexed:true) and highlighting (field_stored:true, highlights:... ) number fields (e.g. field_type="I64") fixed

### Changed

- REST API examples in test_api.rest updated

## [0.1.11] - 2024-03-24

### Added

- get_document REST API endpoint with highlights and fields parameter implemented

### Changed

- REST API examples in test_api.rest updated
- Increased REST API robustness
- Server documentation for REST API endpoints improved (readme + doc)
- REST API query index via GET now supports alternatively url query parameter (only query,offst,lengh,realtime) or JSON request object (all parameters)

## [0.1.10] - 2024-03-21

### Fixed

- Realtime search fixed
- REST API examples fixed

## [0.1.9] - 2024-03-20

### Added

- Highlighter parameter added to get_document.
highlighter generates fragments (snippets, summaries) from each specified field to provide a "keyword in context" (KWIC) functionality.
With highlight_markup the matching query terms within the fragments can be highlighted with HTML markup.
- Fields parameter added to get_document.
fields allows to specify at query time which stored fields to return 

### Fixed

- Server REST API exception handling fixed
- REST API examples fixed

### Changed

- More verbose REST API status message
- Documentation updated

### Removed
