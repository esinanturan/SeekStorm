use std::{
    collections::HashMap,
    env::current_exe,
    fs::{self, File},
    path::{Path, PathBuf},
    time::Instant,
};

use ahash::AHashMap;
use itertools::Itertools;
use serde_json::Value;
use std::collections::HashSet;
use utoipa::{OpenApi, ToSchema};

use seekstorm::{
    commit::Commit,
    highlighter::{Highlight, highlighter},
    index::{
        AccessType, Close, Clustering, DeleteDocument, DeleteDocuments, DeleteDocumentsByQuery,
        DistanceField, Document, DocumentCompression, Facet, FileType, FrequentwordType,
        IS_SYSTEM_LE, IndexArc, IndexDocument, IndexDocuments, IndexMetaObject, LexicalSimilarity,
        MinMaxFieldJson, NgramSet, QueryCompletion, SchemaField, SpellingCorrection, StemmerType,
        StopwordType, Synonym, TokenizerType, UpdateDocument, UpdateDocuments, create_index,
        open_index,
    },
    ingest::IndexPdfBytes,
    iterator::{GetIterator, IteratorResult},
    search::{
        FacetFilter, QueryFacet, QueryRewriting, QueryType, ResultSort, ResultType, Search,
        SearchMode,
    },
    utils::decode_bytes_from_base64_string,
    vector::Inference,
};
use serde::{Deserialize, Serialize};

use crate::{
    VERSION,
    http_server::calculate_hash,
    multi_tenancy::{ApikeyObject, ApikeyQuotaObject},
};

const APIKEY_PATH: &str = "apikey.json";

/// Search request object
#[derive(Deserialize, Serialize, Clone, ToSchema, Debug)]
pub struct SearchRequestObject {
    /// Query string, search operators + - "" are recognized.
    #[serde(rename = "query")]
    pub query_string: String,
    /// Optional query vector: If None, then the query vector is derived from the query string using the specified model. If Some, then the query vector is used for semantic search and the query string is only used for lexical search and highlighting.
    #[serde(default)]
    pub query_vector: Option<Value>,
    #[serde(default)]
    #[schema(required = false, default = false, example = false)]
    /// Enable empty query: if true, an empty query string iterates through all indexed documents, supporting the query parameters: offset, length, query_facets, facet_filter, result_sort,
    /// otherwise an empty query string returns no results.
    /// Typical use cases include index browsing, index export, conversion, analytics, audits, and inspection.
    pub enable_empty_query: bool,
    #[serde(default)]
    #[schema(required = false, minimum = 0, default = 0, example = 0)]
    /// Offset of search results to return.
    pub offset: usize,
    /// Number of search results to return.
    #[serde(default = "length_api")]
    #[schema(required = false, minimum = 1, default = 10, example = 10)]
    pub length: usize,
    #[serde(default)]
    pub result_type: ResultType,
    /// True realtime search: include indexed, but uncommitted documents into search results.
    #[serde(default)]
    pub realtime: bool,
    /// Specify field names where to create keyword-in-context fragments and highlight query terms.
    #[serde(default)]
    pub highlights: Vec<Highlight>,
    /// Specify field names where to search at querytime, whereas SchemaField.indexed is set at indextime. If empty then all indexed fields are searched.
    #[schema(required = false, example = json!(["title"]))]
    #[serde(default)]
    pub field_filter: Vec<String>,
    /// Specify names of fields to return in the search results, where SchemaField.store is set at indextime. If empty then all stored fields are returned.
    #[serde(default)]
    pub fields: Vec<String>,
    /// Specify distance fields to derive at query time and return in the search results.
    #[serde(default)]
    pub distance_fields: Vec<DistanceField>,
    /// Facets to return with search results: if empty then no facets are returned. Facets are only enabled on facet fields that are defined in schema at create_index!
    #[serde(default)]
    pub query_facets: Vec<QueryFacet>,
    /// Facet filters to filter search results by facet values: if empty then no facet filters are applied. Facet filters are only enabled on facet fields that are defined in schema at create_index!
    #[serde(default)]
    pub facet_filter: Vec<FacetFilter>,
    /// Sort field and order:
    /// Search results are sorted by the specified facet field, either in ascending or descending order.
    /// If no sort field is specified, then the search results are sorted by rank in descending order per default.
    /// Multiple sort fields are combined by a "sort by, then sort by"-method ("tie-breaking"-algorithm).
    /// The results are sorted by the first field, and only for those results where the first field value is identical (tie) the results are sub-sorted by the second field,
    /// until the n-th field value is either not equal or the last field is reached.
    /// A special _score field (BM25x), reflecting how relevant the result is for a given search query (phrase match, match in title etc.) can be combined with any of the other sort fields as primary, secondary or n-th search criterium.
    /// Sort is only enabled on facet fields that are defined in schema at create_index!
    /// Examples:
    /// - result_sort = vec![ResultSort {field: "price".into(), order: SortOrder::Descending, base: FacetValue::None},ResultSort {field: "language".into(), order: SortOrder::Ascending, base: FacetValue::None}];
    /// - result_sort = vec![ResultSort {field: "location".into(),order: SortOrder::Ascending, base: FacetValue::Point(vec![38.8951, -77.0364])}];
    #[schema(required = false, example = json!([{"field": "date", "order": "Ascending", "base": "None" }]))]
    #[serde(default)]
    pub result_sort: Vec<ResultSort>,
    /// Specify default query type: (default=Intersection). This can be overwritten by search operator within the query string (+-"").
    #[schema(required = false, example = QueryType::Intersection)]
    #[serde(default = "query_type_api")]
    pub query_type_default: QueryType,
    /// Specify query rewriting method for search query correction and completion: (default=SearchOnly).
    #[schema(required = false, example = QueryRewriting::SearchOnly)]
    #[serde(default = "query_rewriting_api")]
    pub query_rewriting: QueryRewriting,
    /// Specify search mode: (default=Lexical).
    #[schema(required = false, example = SearchMode::Lexical)]
    #[serde(default = "search_mode_api")]
    pub search_mode: SearchMode,
}

fn search_mode_api() -> SearchMode {
    SearchMode::Lexical
}

fn query_type_api() -> QueryType {
    QueryType::Intersection
}

fn query_rewriting_api() -> QueryRewriting {
    QueryRewriting::SearchOnly
}

fn length_api() -> usize {
    10
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SearchResultObject {
    /// Time taken to execute the search query in nanoseconds
    pub time: u128,
    /// Search query string
    pub original_query: String,
    /// Search query string after any automatic query correction or completion
    pub query: String,
    /// Offset of the returned search results
    pub offset: usize,
    /// Number of requested search results
    pub length: usize,
    /// Number of returned search results matching the query
    pub count: usize,
    /// Total number of search results matching the query
    pub count_total: usize,
    /// Vector of search query terms. Can be used e.g. for custom highlighting.
    pub query_terms: Vec<String>,
    #[schema(value_type=Vec<HashMap<String, serde_json::Value>>)]
    /// Vector of search result documents
    pub results: Vec<Document>,
    #[schema(value_type=HashMap<String, Vec<(String, usize)>>)]
    /// Facets with their values and corresponding document counts
    pub facets: AHashMap<String, Facet>,
    /// Suggestions for query correction or completion
    pub suggestions: Vec<String>,
}

/// Create index request object
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CreateIndexRequest {
    /// Index name, used for informational purposes only.
    #[schema(example = "demo_index")]
    pub index_name: String,
    #[schema(required = true, example = json!([
    {"field":"title","field_type":"Text","store":true,"index_lexical":true,"boost":10.0},
    {"field":"body","field_type":"Text","store":true,"index_lexical":true,"longest":true},
    {"field":"url","field_type":"Text","store":true,"index_lexical":false},
    {"field":"date","field_type":"Timestamp","store":true,"index_lexical":false,"facet":true}]))]
    /// Schema definition for the index: field name, field type, and indexing options. The schema defines how documents are indexed and searched. It specifies the fields that are indexed, stored, and used for faceting, as well as the field types and their properties. It also defines whether lexical, hybrid, or vector search is enabled for each field.
    #[serde(default)]
    pub schema: Vec<SchemaField>,
    /// Specify similarity measure for the index: (default=Bm25fProximity). The similarity function is used to calculate the relevance score of search results for a given search query. The choice of similarity function can affect search performance and relevance, depending on the characteristics of the text being indexed and the search queries being executed.
    #[serde(default = "similarity_type_api")]
    pub similarity: LexicalSimilarity,
    /// Specify tokenizer type for the index: (default=UnicodeAlphanumeric). The tokenizer is used to split text into tokens for indexing and searching. The choice of tokenizer can affect search performance and relevance, depending on the language and characteristics of the text being indexed.
    #[serde(default = "tokenizer_type_api")]
    pub tokenizer: TokenizerType,
    #[serde(default)]
    pub stemmer: StemmerType,
    /// Specify stop words for the index. Stop words are not indexed and not searched for. This can be used to reduce index size and improve search performance by excluding high-frequency, low-information terms from the index.
    #[serde(default)]
    pub stop_words: StopwordType,
    /// Specify frequent words for the index. Frequent words are used to optimize search performance for high-frequency terms.
    #[serde(default)]
    pub frequent_words: FrequentwordType,
    /// Specify n-gram indexing for the index. N-gram indexing can improve search performance for certain types of queries.
    /// The n-gram set is defined as a bitwise combination of the following values:
    /// - NgramSet::SingleTerm = 0b00000000,,
    /// - NgramSet::NgramFF = 0b00000001, (Ngram frequent frequent)
    /// - NgramSet::NgramFR = 0b00000010, (Ngram frequent rare)
    /// - NgramSet::NgramRF = 0b00000011, (Ngram rare frequent)
    /// - NgramSet::NgramFFF = 0b00000100, (Ngram frequent frequent frequent)
    /// - NgramSet::NgramRFF = 0b00000101, (Ngram rare frequent frequent)
    /// - NgramSet::NgramFFR = 0b00000110, (Ngram frequent frequent rare)
    /// - NgramSet::NgramFRF = 0b00000111, (Ngram frequent rare frequent)
    /// 
    /// For example, to enable both NgramFF and NgramFFF, set ngram_indexing to 5 (1 | 4).
    /// Note: enabling n-gram indexing (ngram_indexing>0) will increase index size and indexing time, but improves search performance of phrase queries with frequent terms.
    #[serde(default = "ngram_indexing_api")]
    pub ngram_indexing: u8,
    /// Enable document compression for the index. This can reduce the index size on disk and in memory, but may increase indexing and search latency. Default: Snappy compression.
    #[serde(default = "document_compression_api")]
    pub document_compression: DocumentCompression,
    /// Specify synonyms for the index. Synonyms are used to expand search queries with additional terms that have the same or similar meaning, improving recall and search relevance. The multiway option specifies whether the synonym relationship is multiway (if true, all terms in the synonym set are considered synonyms of each other) or one-way (if false, only the first term in the synonym set is considered the main term, and the other terms are considered synonyms of the main term).
    #[schema(required = false, example = json!([{"terms":["berry","lingonberry","blueberry","gooseberry"],"multiway":false}]))]
    #[serde(default)]
    pub synonyms: Vec<Synonym>,
    /// Set number of shards manually or automatically.
    /// - none: number of shards is set automatically = number of physical processor cores (default)
    /// - small: slower indexing, higher latency, slightly higher throughput, faster realtime search, lower RAM consumption
    /// - large: faster indexing, lower latency, slightly lower throughput, slower realtime search, higher RAM consumption
    #[serde(default)]
    pub force_shard_number: Option<usize>,
    /// Enable spelling correction for search queries using the SymSpell algorithm.
    /// When enabled, a SymSpell dictionary is incrementally created during indexing of documents and stored in the index.
    /// In addition you need to set the parameter `query_rewriting` in the search method to enable it per query.
    /// The creation of an individual dictionary derived from the indexed documents improves the correction quality compared to a generic dictionary.
    /// An dictionary per index improves the privacy compared to a global dictionary derived from all indices.
    /// The dictionary is deleted when delete_index or clear_index is called.
    /// Note: enabling spelling correction increases the index size, indexing time and query latency.
    /// Default: None. Enable by setting a value for max_dictionary_edit_distance (1..2 recommended).
    /// The higher the value, the higher the number of errors taht can be corrected - but also the memory consumption, lookup latency, and the number of false positives.
    #[serde(default)]
    pub spelling_correction: Option<SpellingCorrection>,
    /// Enable query completion for search queries using a prefix dictionary. When enabled, a prefix dictionary is incrementally created during indexing of documents and stored in the index. The prefix dictionary is used to generate suggestions for query completion based on the indexed documents. In addition you need to set the parameter `query_rewriting` in the search method to enable it per query. Note: enabling query completion increases the index size, indexing time and query latency.
    #[serde(default)]
    pub query_completion: Option<QueryCompletion>,
    #[serde(default)]
    pub clustering: Clustering,
    /// Enable inference for search and indexing. This can be used to create vector representations of documents and queries for semantic search, e.g. by using a model like PotionBase2M.
    #[serde(default)]
    pub inference: Inference,
}

fn similarity_type_api() -> LexicalSimilarity {
    LexicalSimilarity::Bm25fProximity
}

fn tokenizer_type_api() -> TokenizerType {
    TokenizerType::UnicodeAlphanumeric
}

fn ngram_indexing_api() -> u8 {
    NgramSet::NgramFF as u8 | NgramSet::NgramFFF as u8
}

fn document_compression_api() -> DocumentCompression {
    DocumentCompression::Snappy
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeleteApikeyRequest {
    pub apikey_base64: String,
}

/// Specifies which document ID to return
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct GetIteratorRequest {
    /// base document ID to start the iteration from
    /// Use None to start from the beginning (take>0) or the end (take<0) of the index
    /// In JSON use null for None
    #[serde(default)]
    pub document_id: Option<u64>,
    /// the number of document IDs to skip
    #[serde(default)]
    pub skip: usize,
    /// the number of document IDs to return
    /// take>0: take next t document IDs, take<0: take previous t document IDs
    #[serde(default = "default_1usize")]
    pub take: isize,
    /// if true, also deleted document IDs are included in the result
    #[serde(default)]
    pub include_deleted: bool,
    /// if true, the documents are also retrieved along with their document IDs
    #[serde(default)]
    pub include_document: bool,
    /// which fields to return (if include_document is true, if empty then return all stored fields)
    #[serde(default)]
    pub fields: Vec<String>,
}

fn default_1usize() -> isize {
    1
}

/// Specifies which document and which field to return
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct GetDocumentRequest {
    /// query terms for highlighting
    #[serde(default)]
    pub query_terms: Vec<String>,
    /// which fields to highlight: create keyword-in-context fragments and highlight terms
    #[serde(default)]
    pub highlights: Vec<Highlight>,
    /// which fields to return
    #[serde(default)]
    pub fields: Vec<String>,
    /// which distance fields to derive and return
    #[serde(default)]
    pub distance_fields: Vec<DistanceField>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct IndexResponseObject {
    /// Index ID
    pub id: u64,
    /// Index name
    #[schema(example = "demo_index")]
    pub name: String,
    #[schema(example = json!({
        "title":{
            "field":"title",
            "store":true,
            "index_lexical":true,
            "field_type":"Text",
            "boost":10.0,
            "field_id":0
        },
        "body":{
            "field":"body",
            "store":true,
            "index_lexical":true,
            "field_type":"Text",
            "field_id":1
        },
        "url":{
           "field":"url",
           "store":true,
           "index_lexical":false,
           "field_type":"Text",
           "field_id":2
        },
        "date":{
           "field":"date",
           "store":true,
           "index_lexical":false,
           "field_type":"Timestamp",
           "facet":true,
           "field_id":3
        }
     }))]
    pub schema: HashMap<String, SchemaField>,
    /// Number of indexed documents
    pub indexed_doc_count: usize,
    /// Number of committed documents
    pub committed_doc_count: usize,
    /// Number of operations: index, update, delete, queries
    pub operations_count: u64,
    /// Number of queries, for quotas and billing
    pub query_count: u64,
    /// SeekStorm version the index was created with
    #[schema(example = "0.11.1")]
    pub version: String,
    /// Minimum and maximum values of numeric facet fields
    #[schema(example = json!({"date":{"min":831306011,"max":1730901447}}))]
    pub facets_minmax: HashMap<String, MinMaxFieldJson>,
}

/// Save file atomically
pub(crate) fn save_file_atomically(path: &PathBuf, content: String) {
    let mut temp_path = path.clone();
    temp_path.set_extension("bak");
    fs::write(&temp_path, content).unwrap();
    match fs::rename(temp_path, path) {
        Ok(_) => {}
        Err(e) => println!("error: {e:?}"),
    }
}

pub(crate) fn save_apikey_data(apikey: &ApikeyObject, index_path: &PathBuf) {
    let apikey_id: u64 = apikey.id;

    let apikey_id_path = Path::new(&index_path).join(apikey_id.to_string());
    let apikey_persistence_json = serde_json::to_string(&apikey).unwrap();
    let apikey_persistence_path = Path::new(&apikey_id_path).join(APIKEY_PATH);
    save_file_atomically(&apikey_persistence_path, apikey_persistence_json);
}

/// Live
///
/// Returns a live message with the SeekStorm server version.
#[utoipa::path(
    tag = "Info",
    get,
    path = "/api/v1/live",
    responses(
        (status = 200, description = "SeekStorm server is live", body = String),
    )
)]
pub(crate) fn live_api() -> String {
    "SeekStorm server ".to_owned() + VERSION
}

/// Create API Key
///
/// Creates an API key and returns the Base64 encoded API key.  
/// Expects the Base64 encoded master API key in the header.  
/// Use the master API key displayed in the server console at startup.
///  
/// WARNING: make sure to set the MASTER_KEY_SECRET environment variable to a secret, otherwise your generated API keys will be compromised.  
/// For development purposes you may also use the SeekStorm server console command 'create' to create an demo API key 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='.
#[utoipa::path(
    tag = "API Key",
    post,
    path = "/api/v1/apikey",
    params(
        ("apikey" = String, Header, description = "YOUR_MASTER_API_KEY",example="BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="),
    ),
    request_body = inline(ApikeyQuotaObject),
    responses(
        (status = 200, description = "API key created, returns Base64 encoded API key", body = String),
        (status = UNAUTHORIZED, description = "master_apikey invalid"),
        (status = UNAUTHORIZED, description = "master_apikey missing")
    )
)]
pub(crate) fn create_apikey_api<'a>(
    index_path: &'a PathBuf,
    apikey_quota_request_object: ApikeyQuotaObject,
    apikey: &[u8],
    apikey_list: &'a mut HashMap<u128, ApikeyObject>,
) -> &'a mut ApikeyObject {
    let apikey_hash_u128 = calculate_hash(&apikey) as u128;

    let mut apikey_id: u64 = 0;
    let mut apikey_list_vec: Vec<(&u128, &ApikeyObject)> = apikey_list.iter().collect();
    apikey_list_vec.sort_by_key(|a| a.1.id);
    for value in apikey_list_vec {
        if value.1.id == apikey_id {
            apikey_id = value.1.id + 1;
        } else {
            break;
        }
    }

    let apikey_object = ApikeyObject {
        id: apikey_id,
        apikey_hash: apikey_hash_u128,
        quota: apikey_quota_request_object,
        index_list: HashMap::new(),
    };

    let apikey_id_path = Path::new(&index_path).join(apikey_id.to_string());
    fs::create_dir_all(apikey_id_path).unwrap();

    save_apikey_data(&apikey_object, index_path);

    apikey_list.insert(apikey_hash_u128, apikey_object);
    apikey_list.get_mut(&apikey_hash_u128).unwrap()
}

/// Delete API Key
///
/// Deletes an API and returns the number of remaining API keys.
/// Expects the Base64 encoded master API key in the header.
/// WARNING: This will delete all indices and documents associated with the API key.
#[utoipa::path(
    delete,
    tag = "API Key",
    path = "/api/v1/apikey",
    params(
        ("apikey" = String, Header, description = "YOUR_MASTER_API_KEY",example="BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="),
    ),
    responses(
        (status = 200, description = "API key deleted, returns number of remaining API keys", body = u64),
        (status = UNAUTHORIZED, description = "master_apikey invalid"),
        (status = UNAUTHORIZED, description = "master_apikey missing")
    )
)]
pub(crate) fn delete_apikey_api(
    index_path: &PathBuf,
    apikey_list: &mut HashMap<u128, ApikeyObject>,
    apikey_hash: u128,
) -> Result<u64, String> {
    if let Some(apikey_object) = apikey_list.get(&apikey_hash) {
        let apikey_id_path = Path::new(&index_path).join(apikey_object.id.to_string());
        println!("delete path {}", apikey_id_path.to_string_lossy());
        fs::remove_dir_all(&apikey_id_path).unwrap();

        apikey_list.remove(&apikey_hash);
        Ok(apikey_list.len() as u64)
    } else {
        Err("not found".to_string())
    }
}

/// Open all indices below a single apikey
pub(crate) async fn open_all_indices(
    index_path: &PathBuf,
    index_list: &mut HashMap<u64, IndexArc>,
) {
    if !Path::exists(index_path) {
        fs::create_dir_all(index_path).unwrap();
    }

    for result in fs::read_dir(index_path).unwrap() {
        let path = result.unwrap();
        if path.path().is_dir() {
            let single_index_path = path.path();

            let index_arc = match open_index(&single_index_path).await {
                Ok(index_arc) => index_arc,
                Err(err) => {
                    println!("{} {}", err, single_index_path.display());
                    continue;
                }
            };

            let index_id = index_arc.read().await.meta.id;

            index_list.insert(index_id, index_arc);
        }
    }
}

/// Open api key
pub(crate) async fn open_apikey(
    index_path: &PathBuf,
    apikey_list: &mut HashMap<u128, ApikeyObject>,
) -> bool {
    let apikey_path = Path::new(&index_path).join(APIKEY_PATH);
    match fs::read_to_string(apikey_path) {
        Ok(apikey_string) => {
            let mut apikey_object: ApikeyObject = serde_json::from_str(&apikey_string).unwrap();

            open_all_indices(index_path, &mut apikey_object.index_list).await;
            apikey_list.insert(apikey_object.apikey_hash, apikey_object);

            true
        }
        Err(_) => false,
    }
}

/// Open all apikeys in the specified path
pub(crate) async fn open_all_apikeys(
    index_path: &PathBuf,
    apikey_list: &mut HashMap<u128, ApikeyObject>,
) -> bool {
    let mut test_index_flag = false;
    if !Path::exists(index_path) {
        println!("index path not found: {} ", index_path.to_string_lossy());
        fs::create_dir_all(index_path).unwrap();
    }

    for result in fs::read_dir(index_path).unwrap() {
        let path = result.unwrap();
        if path.path().is_dir() {
            let single_index_path = path.path();
            test_index_flag |= open_apikey(&single_index_path, apikey_list).await;
        }
    }
    test_index_flag
}

/// Create Index
///
/// Create an index within the directory associated with the specified API key and return the index_id.
#[utoipa::path(
    post,
    tag = "Index",
    path = "/api/v1/index",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    ),


    request_body(
        content = CreateIndexRequest,
        examples(
            ("Example: Lexical index" = (value = json!({
				"schema":[{
					"field": "title", 
					"field_type": "Text", 
					"store": true, 
					"index_lexical": true,
					"boost":10.0
				},
					{
					"field": "body",
					"field_type": "Text", 
					"store": true, 
					"index_lexical": true,
					"longest": true
				},
				{
					"field": "url", 
					"field_type": "String32", 
					"store": true, 
					"index_lexical": false
				}],
				"index_name": "wikipedia",
				"similarity": "Bm25fProximity",
				"tokenizer": "UnicodeAlphanumeric"
			}))),
			("Example: Hybrid index" = (value = json!({
				"schema":[{
					"field": "title", 
					"field_type": "Text", 
					"store": true, 
					"index_lexical": true,
					"index_vector": true
				},
				{
					"field": "body",
					"field_type": "Text", 
					"store": true, 
					"index_lexical": true,
					"index_vector": true
				},
					{
					"field": "url", 
					"field_type": "String32", 
					"store": true, 
					"index_lexical": false,
					"index_vector": false
				}],
				"index_name": "wikipedia",
				"similarity": "Bm25fProximity",
				"tokenizer": "UnicodeAlphanumeric",
				"clustering": "Auto",
				"inference": {"Model2Vec": { "model": "PotionBase2M", "chunk_size": 1000, "quantization": "ScalarQuantizationI8" }}
			}))),
            ("Example: Vector index" = (value = json!({
				"schema":[
				{
					"field":"vector",
					"field_type":"Json",
					"store":false,
					"index_lexical":false,
					"index_vector":true
				},
				{
					"field":"index",
					"field_type":"Text",
					"store":true,
					"index_lexical":false,
					"index_vector":false
				}],
				"index_name": "sift1m",
				"clustering": "Auto",
				"inference": {
					"External": { "dimensions": 128, "precision": "F32", "quantization": "ScalarQuantizationI8", "similarity": "Euclidean" }
				}
			})))
        )
    ),

    responses(
        (status = OK, description = "Index created, returns the index_id", body = u64),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "API key does not exists"),
        (status = UNAUTHORIZED, description = "API key is missing"),
        (status = UNAUTHORIZED, description = "API key does not exists")
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn create_index_api<'a>(
    index_path: &'a PathBuf,
    index_name: String,
    schema: Vec<SchemaField>,
    lexical_similarity: LexicalSimilarity,
    tokenizer: TokenizerType,
    stemmer: StemmerType,
    stop_words: StopwordType,
    frequent_words: FrequentwordType,
    ngram_indexing: u8,
    document_compression: DocumentCompression,
    synonyms: Vec<Synonym>,
    force_shard_number: Option<usize>,
    apikey_object: &'a mut ApikeyObject,
    spelling_correction: Option<SpellingCorrection>,
    query_completion: Option<QueryCompletion>,
    mute: bool,
    clustering: Clustering,
    inference: Inference,
) -> u64 {
    let mut index_id: u64 = 0;
    for id in apikey_object.index_list.keys().sorted() {
        if *id == index_id {
            index_id = id + 1;
        } else {
            break;
        }
    }

    let index_id_path = Path::new(&index_path)
        .join(apikey_object.id.to_string())
        .join(index_id.to_string());
    fs::create_dir_all(&index_id_path).unwrap();

    let meta = IndexMetaObject {
        id: index_id,
        name: index_name,
        lexical_similarity,
        tokenizer,
        stemmer,
        stop_words,
        frequent_words,
        ngram_indexing,
        document_compression,
        access_type: AccessType::Mmap,
        spelling_correction,
        query_completion,
        clustering,
        inference,
    };

    let index_arc = create_index(
        &index_id_path,
        meta,
        &schema,
        &synonyms,
        11,
        mute,
        force_shard_number,
    )
    .await
    .unwrap();

    apikey_object.index_list.insert(index_id, index_arc);

    index_id
}

/// Delete Index
///
/// Delete an index within the directory associated with the specified API key and return the number of remaining indices.
#[utoipa::path(
    delete,
    tag = "Index",
    path = "/api/v1/index/{index_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    responses(
        (status = 200, description = "Index deleted, returns the number of indices", body = u64),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = NOT_FOUND, description = "Index_id does not exists"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing")
    )
)]
pub(crate) async fn delete_index_api(
    index_id: u64,
    index_list: &mut HashMap<u64, IndexArc>,
) -> Result<u64, String> {
    if let Some(index_arc) = index_list.get(&index_id) {
        let mut index_mut = index_arc.write().await;
        index_mut.delete_index();
        drop(index_mut);
        index_list.remove(&index_id);

        Ok(index_list.len() as u64)
    } else {
        Err("index_id not found".to_string())
    }
}

/// Commit Index
///
/// Commit moves indexed documents from the intermediate uncompressed data structure (array lists/HashMap, queryable by realtime search) in RAM
/// to the final compressed data structure (roaring bitmap) on Mmap or disk -
/// which is persistent, more compact, with lower query latency and allows search with realtime=false.
/// Commit is invoked automatically each time 64K documents are newly indexed **per shard** as well as on close_index (e.g. server quit).
/// There is no way to prevent this automatic commit by not manually invoking it.
/// But commit can also be invoked manually at any time at any number of newly indexed documents.
/// commit is a **hard commit** for persistence on disk. A **soft commit** for searchability
/// is invoked implicitly with every index_doc,
/// i.e. the document can immediately searched and included in the search results
/// if it matches the query AND the query parameter realtime=true is enabled.
/// **Use commit with caution, as it is an expensive operation**.
/// **Usually, there is no need to invoke it manually**, as it is invoked automatically every 64k documents **per shard** and when the index is closed with close_index.
/// Before terminating the program, always call close_index (commit), otherwise all documents indexed since last (manual or automatic) commit are lost.
/// There are only 2 reasons that justify a manual commit:
/// 1. if you want to search newly indexed documents without using realtime=true for search performance reasons or
/// 2. if after indexing new documents there won't be more documents indexed (for some time),
///    so there won't be (soon) a commit invoked automatically at the next 64k threshold **per shard** or close_index,
///    but you still need immediate persistence guarantees on disk to protect against data loss in the event of a crash.
#[utoipa::path(
    patch,
    tag = "Index",
    path = "/api/v1/index/{index_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    responses(
        (status = 200, description = "Index committed, returns the number of committed documents", body = u64),
        (status = BAD_REQUEST, description = "Index id invalid or missing"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing")
    )
)]
pub(crate) async fn commit_index_api(index_arc: &IndexArc) -> Result<u64, String> {
    let index_arc_clone = index_arc.clone();
    let index_ref = index_arc.read().await;
    let indexed_doc_count = index_ref.indexed_doc_count().await;

    drop(index_ref);
    index_arc_clone.commit().await;

    Ok(indexed_doc_count as u64)
}

pub(crate) async fn close_index_api(index_arc: &IndexArc) -> Result<u64, String> {
    let indexed_doc_count = index_arc.read().await.indexed_doc_count().await;
    index_arc.close().await;

    Ok(indexed_doc_count as u64)
}

pub(crate) async fn set_synonyms_api(
    index_arc: &IndexArc,
    synonyms: Vec<Synonym>,
) -> Result<usize, String> {
    let mut index_mut = index_arc.write().await;
    index_mut.set_synonyms(&synonyms)
}

pub(crate) async fn add_synonyms_api(
    index_arc: &IndexArc,
    synonyms: Vec<Synonym>,
) -> Result<usize, String> {
    let mut index_mut = index_arc.write().await;
    index_mut.add_synonyms(&synonyms)
}

pub(crate) async fn get_synonyms_api(index_arc: &IndexArc) -> Result<Vec<Synonym>, String> {
    let index_ref = index_arc.read().await;
    index_ref.get_synonyms()
}

/// Get Index Info
///
/// Get index Info from index with index_id
#[utoipa::path(
    get,
    tag = "Index",
    path = "/api/v1/index/{index_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    responses(
        (
            status = 200, description = "Index found, returns the index info", 
            body = IndexResponseObject,
        ),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn get_index_info_api(
    index_id: u64,
    index_list: &HashMap<u64, IndexArc>,
) -> Result<IndexResponseObject, String> {
    if let Some(index_arc) = index_list.get(&index_id) {
        let index_ref = index_arc.read().await;

        Ok(IndexResponseObject {
            version: VERSION.to_string(),
            schema: index_ref.schema_map.clone(),
            id: index_ref.meta.id,
            name: index_ref.meta.name.clone(),
            indexed_doc_count: index_ref.indexed_doc_count().await,
            committed_doc_count: index_ref.committed_doc_count().await,
            operations_count: 0,
            query_count: 0,
            facets_minmax: index_ref.index_facets_minmax().await,
        })
    } else {
        Err("index_id not found".to_string())
    }
}

/// Get API Key Info
///
/// Get info about all indices associated with the specified API key
#[utoipa::path(
    get,
    tag = "API Key",
    path = "/api/v1/apikey",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    ),
    responses(
        (
            status = 200, description = "Indices found, returns a list of index info", 
            body = Vec<IndexResponseObject>,
        ),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index ID or API key missing"),
        (status = UNAUTHORIZED, description = "API key does not exists"),
    )
)]
pub(crate) async fn get_apikey_indices_info_api(
    index_list: &HashMap<u64, IndexArc>,
) -> Result<Vec<IndexResponseObject>, String> {
    let mut index_response_object_vec: Vec<IndexResponseObject> = Vec::new();
    for index in index_list.iter() {
        let index_ref = index.1.read().await;

        index_response_object_vec.push(IndexResponseObject {
            version: VERSION.to_string(),
            schema: index_ref.schema_map.clone(),
            id: index_ref.meta.id,
            name: index_ref.meta.name.clone(),
            indexed_doc_count: index_ref.indexed_doc_count().await,
            committed_doc_count: index_ref.committed_doc_count().await,
            operations_count: 0,
            query_count: 0,
            facets_minmax: index_ref.index_facets_minmax().await,
        });
    }

    Ok(index_response_object_vec)
}

/// Index Document(s)
///
/// Index a JSON document or an array of JSON documents (bulk), each consisting of arbitrary key-value pairs to the index with the specified apikey and index_id, and return the number of indexed docs.
/// Index documents enables true real-time search (as opposed to near realtime.search):
/// When in query_index the parameter `realtime` is set to `true` then indexed, but uncommitted documents are immediately included in the search results, without requiring a commit or refresh.
/// Therefore a explicit commit_index is almost never required, as it is invoked automatically after 64k documents are indexed **per shard** or on close_index for persistence.
#[utoipa::path(
    post,
    tag = "Document",
    path = "/api/v1/index/{index_id}/doc",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),


 	request_body(
        content = HashMap<String, Value>, description = "JSON document or array of JSON documents, each consisting of key-value pairs",content_type = "application/json", 
        examples(
    ("Example: Single Lexical/Hybrid document" = (value = json!(  {
        "title": "title1 test",
        "body": "body1",
        "url": "url1"
		} ))),
	("Example: Multiple Lexical/Hybrid documents" = (value = json!(  [
{
    "title":"title2",
    "body":"body2 test",
    "url":"url2"
},
{
    "title":"title3 test",
    "body":"body3 test",
    "url":"url3"
}
]   ))),
    ("Example: Multiple Vector documents" = (value = json!(  [
    {"vector":[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.070, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.080, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.090, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.100, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.120, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128],"index":"0"},
    {"vector":[0.129, 0.130, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139, 0.140, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149, 0.150, 0.151, 0.152, 0.153, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.160, 0.161, 0.162, 0.163, 0.164, 0.165, 0.166, 0.167, 0.168, 0.169, 0.170, 0.171, 0.172, 0.173, 0.174, 0.175, 0.176, 0.177, 0.178, 0.179, 0.180, 0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.190, 0.191, 0.192, 0.193, 0.194, 0.195, 0.196, 0.197, 0.198, 0.199, 0.200, 0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.210, 0.211, 0.212, 0.213, 0.214, 0.215, 0.216, 0.217, 0.218, 0.219, 0.220, 0.221, 0.222, 0.223, 0.224, 0.225, 0.226, 0.227, 0.228, 0.229, 0.230, 0.231, 0.232, 0.233, 0.234, 0.235, 0.236, 0.237, 0.238, 0.239, 0.240, 0.241, 0.242, 0.243, 0.244, 0.245, 0.246, 0.247, 0.248, 0.249, 0.250, 0.251, 0.252, 0.253, 0.254, 0.255, 0.256],"index":"1"},
    {"vector":[0.257, 0.258, 0.259, 0.260, 0.261, 0.262, 0.263, 0.264, 0.265, 0.266, 0.267, 0.268, 0.269, 0.270, 0.271, 0.272, 0.273, 0.274, 0.275, 0.276, 0.277, 0.278, 0.279, 0.280, 0.281, 0.282, 0.283, 0.284, 0.285, 0.286, 0.287, 0.288, 0.289, 0.290, 0.291, 0.292, 0.293, 0.294, 0.295, 0.296, 0.297, 0.298, 0.299, 0.300, 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.310, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317, 0.318, 0.319, 0.320, 0.321, 0.322, 0.323, 0.324, 0.325, 0.326, 0.327, 0.328, 0.329, 0.330, 0.331, 0.332, 0.333, 0.334, 0.335, 0.336, 0.337, 0.338, 0.339, 0.340, 0.341, 0.342, 0.343, 0.344, 0.345, 0.346, 0.347, 0.348, 0.349, 0.350, 0.351, 0.352, 0.353, 0.354, 0.355, 0.356, 0.357, 0.358, 0.359, 0.360, 0.361, 0.362, 0.363, 0.364, 0.365, 0.366, 0.367, 0.368, 0.369, 0.370, 0.371, 0.372, 0.373, 0.374, 0.375, 0.376, 0.377, 0.378, 0.379, 0.380, 0.381, 0.382, 0.383, 0.384],"index":"2"}
]   )))
        )
    ),
    responses(
        (status = 200, description = "Document indexed, returns the number of indexed documents", body = usize),
        (status = BAD_REQUEST, description = "Document object invalid"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing")
    )
)]
pub(crate) async fn index_document_api(
    index_arc: &IndexArc,
    document: Document,
) -> Result<usize, String> {
    index_arc.index_document(document, FileType::None).await;

    Ok(index_arc.read().await.indexed_doc_count().await)
}

/// Index PDF file
///
/// Index PDF file (byte array) to the index with the specified apikey and index_id, and return the number of indexed docs.
/// - Converts PDF to a JSON document with "title", "body", "url" and "date" fields and indexes it.
/// - extracts title from metatag, or first line of text, or from filename
/// - extracts creation date from metatag, or from file creation date (Unix timestamp: the number of seconds since 1 January 1970)
/// - copies all ingested pdf files to "files" subdirectory in index
#[utoipa::path(
    post,
    tag = "PDF File",
    path = "/api/v1/index/{index_id}/file",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("file" = String, Header, description = "filepath from header for JSON 'url' field"),
        ("date" = String, Header, description = "date (timestamp) from header, as fallback for JSON 'date' field, if PDF date meta tag unaivailable"),
        ("index_id" = u64, Path, description = "index id"),
    ),
    request_body = inline(&[u8]),
    responses(
        (status = 200, description = "PDF file indexed, returns the number of indexed documents", body = usize),
        (status = BAD_REQUEST, description = "Document object invalid"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing")
    )
)]
pub(crate) async fn index_file_api(
    index_arc: &IndexArc,
    file_path: &Path,
    file_date: i64,
    document: &[u8],
) -> Result<usize, String> {
    match index_arc
        .index_pdf_bytes(file_path, file_date, document)
        .await
    {
        Ok(_) => Ok(index_arc.read().await.indexed_doc_count().await),
        Err(e) => Err(e),
    }
}

/// Get PDF file
///
/// Get PDF file from index with index_id
/// ⚠️ Use search or get_iterator first to obtain s valid doc_id. Document IDs are not guaranteed to be continuous and gapless!
#[utoipa::path(
    get,
    tag = "PDF File",
    path = "/api/v1/index/{index_id}/file/{document_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
        ("document_id" = u64, Path, description = "document id"),
    ),
    responses(
        (status = 200, description = "PDF file found, returns the PDF file as byte array", body = [u8]),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "doc_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "Document id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn get_file_api(index_arc: &IndexArc, document_id: usize) -> Option<Vec<u8>> {
    if !index_arc.read().await.stored_field_names.is_empty() {
        index_arc.read().await.get_file(document_id).await.ok()
    } else {
        None
    }
}

pub(crate) async fn index_documents_api(
    index_arc: &IndexArc,
    document_vec: Vec<Document>,
) -> Result<usize, String> {
    index_arc.index_documents(document_vec).await;
    Ok(index_arc.read().await.indexed_doc_count().await)
}

/// Get Document
///
/// Get document from index with index_id
/// ⚠️ Use search or get_iterator first to obtain a valid doc_id. Document IDs are not guaranteed to be continuous and gapless!
#[utoipa::path(
    get,
    tag = "Document",
    path = "/api/v1/index/{index_id}/doc/{document_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
        ("document_id" = u64, Path, description = "document id"),
    ),
    request_body(content = GetDocumentRequest, example=json!({
        "query_terms": ["test"],
        "fields": ["title", "body"],
        "highlights": [
        { "field": "title", "fragment_number": 0, "fragment_size": 1000, "highlight_markup": true},
        { "field": "body", "fragment_number": 2, "fragment_size": 160, "highlight_markup": true},
        { "field": "body", "name": "body2", "fragment_number": 0, "fragment_size": 4000, "highlight_markup": true}]
    })),
    responses(
        (status = 200, description = "Document found, returns the JSON document consisting of arbitrary key-value pairs", body = HashMap<String, Value>),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "doc_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "Document id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn get_document_api(
    index_arc: &IndexArc,
    document_id: usize,
    get_document_request: GetDocumentRequest,
) -> Option<Document> {
    if !index_arc.read().await.stored_field_names.is_empty() {
        let highlighter_option = if get_document_request.highlights.is_empty()
            || get_document_request.query_terms.is_empty()
        {
            None
        } else {
            Some(
                highlighter(
                    index_arc,
                    get_document_request.highlights,
                    get_document_request.query_terms,
                )
                .await,
            )
        };

        index_arc
            .read()
            .await
            .get_document(
                document_id,
                true,
                &highlighter_option,
                &HashSet::from_iter(get_document_request.fields),
                &get_document_request.distance_fields,
            )
            .await
            .ok()
    } else {
        None
    }
}

/// Update Document(s)
///
/// Update a JSON document or an array of JSON documents (bulk), each consisting of arbitrary key-value pairs to the index with the specified apikey and index_id, and return the number of indexed docs.
/// Update document is a combination of delete_document and index_document.
/// All current limitations of delete_document apply.
/// Update documents enables true real-time search (as opposed to near realtime.search):
/// When in query_index the parameter `realtime` is set to `true` then indexed, but uncommitted documents are immediately included in the search results, without requiring a commit or refresh.
/// Therefore a explicit commit_index is almost never required, as it is invoked automatically after 64k documents are indexed **per shard** or on close_index for persistence.
#[utoipa::path(
    patch,
    tag = "Document",
    path = "/api/v1/index/{index_id}/doc",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    request_body(content = (u64, HashMap<String, Value>), description = "Tuple of (doc_id, JSON document) or array of tuples (doc_id, JSON documents), each JSON document consisting of arbitrary key-value pairs", content_type = "application/json", example=json!([0,{"title":"title1 test","body":"body1","url":"url1"}])),
    responses(
        (status = 200, description = "Document indexed, returns the number of indexed documents", body = usize),
        (status = BAD_REQUEST, description = "Document object invalid"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing")
    )
)]
pub(crate) async fn update_document_api(
    index_arc: &IndexArc,
    id_document: (u64, Document),
) -> Result<u64, String> {
    index_arc.update_document(id_document).await;
    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

pub(crate) async fn update_documents_api(
    index_arc: &IndexArc,
    id_document_vec: Vec<(u64, Document)>,
) -> Result<u64, String> {
    index_arc.update_documents(id_document_vec).await;
    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

/// Delete Document
///
/// Delete document by document_id from index with index_id
/// ⚠️ Use search or get_iterator first to obtain a valid doc_id. Document IDs are not guaranteed to be continuous and gapless!
/// Immediately effective, indpendent of commit.
/// Index space used by deleted documents is not reclaimed (until compaction is implemented), but result_count_total is updated.
/// By manually deleting the delete.bin file the deleted documents can be recovered (until compaction).
/// Deleted documents impact performance, especially but not limited to counting (Count, TopKCount). They also increase the size of the index (until compaction is implemented).
/// For minimal query latency delete index and reindexing documents is preferred over deleting documents (until compaction is implemented).
/// BM25 scores are not updated (until compaction is implemented), but the impact is minimal.
#[utoipa::path(
    delete,
    tag = "Document",
    path = "/api/v1/index/{index_id}/doc/{document_id}",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
        ("document_id" = u64, Path, description = "document id"),
    ),
    responses(
        (status = 200, description = "Document deleted, returns indexed documents count", body = usize),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "doc_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "Document id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn delete_document_by_parameter_api(
    index_arc: &IndexArc,
    document_id: u64,
) -> Result<u64, String> {
    index_arc.delete_document(document_id).await;
    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

/// Delete Document(s) by Request Object
///
/// Delete document by document_id, by array of document_id (bulk), by query (SearchRequestObject) from index with index_id, or clear all documents from index.
/// Immediately effective, indpendent of commit.
/// Index space used by deleted documents is not reclaimed (until compaction is implemented), but result_count_total is updated.
/// By manually deleting the delete.bin file the deleted documents can be recovered (until compaction).
/// Deleted documents impact performance, especially but not limited to counting (Count, TopKCount). They also increase the size of the index (until compaction is implemented).
/// For minimal query latency delete index and reindexing documents is preferred over deleting documents (until compaction is implemented).
/// BM25 scores are not updated (until compaction is implemented), but the impact is minimal.
/// Document ID can by obtained by search. When deleting by query (SearchRequestObject), it is advised to perform a dry run search first, to see which documents will be deleted.
#[utoipa::path(
    delete,
    tag = "Document",
    path = "/api/v1/index/{index_id}/doc",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    request_body(content = SearchRequestObject, description = "Specifies the document(s) to delete by different request objects\n- 'clear' : delete all documents in index (clear index)\n- u64 : delete single doc ID\n- [u64] : delete array of doc ID \n- SearchRequestObject : delete documents by query", content_type = "application/json", example=json!({
        "query":"test",
        "offset":0,
        "length":10,
        "realtime": false,
        "field_filter": ["title", "body"]
    })),

    responses(
        (status = 200, description = "Document deleted, returns indexed documents count", body = usize),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "doc_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "Document id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn delete_document_by_object_api(
    index_arc: &IndexArc,
    document_id: u64,
) -> Result<u64, String> {
    index_arc.delete_document(document_id).await;
    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

pub(crate) async fn delete_documents_by_object_api(
    index_arc: &IndexArc,
    document_id_vec: Vec<u64>,
) -> Result<u64, String> {
    index_arc.delete_documents(document_id_vec).await;
    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

pub(crate) async fn delete_documents_by_query_api(
    index_arc: &IndexArc,
    search_request: SearchRequestObject,
) -> Result<u64, String> {
    index_arc
        .delete_documents_by_query(
            search_request.query_string.to_owned(),
            search_request.query_type_default,
            search_request.offset,
            search_request.length,
            search_request.realtime,
            search_request.field_filter,
            search_request.facet_filter,
            search_request.result_sort,
        )
        .await;

    Ok(index_arc.read().await.indexed_doc_count().await as u64)
}

/// Document iterator
///
/// Document iterator via GET and POST are identical, only the way parameters are passed differ.
/// The document iterator allows to iterate over all document IDs and documents in the entire index, forward or backward.
/// It enables efficient sequential access to every document, even in very large indexes, without running a search.
/// Paging through the index works without collecting document IDs to Min-heap in size-limited RAM first.
/// The iterator guarantees that only valid document IDs are returned, even though document IDs are not strictly continuous.
/// Document IDs can also be fetched in batches, reducing round trips and significantly improving performance, especially when using the REST API.
/// Typical use cases include index export, conversion, analytics, audits, and inspection.
/// Explanation of "eventually continuous" docid:
/// In SeekStorm, document IDs become continuous over time. In a multi-sharded index, each shard maintains its own document ID space.
/// Because documents are distributed across shards in a non-deterministic, load-dependent way, shard-local document IDs advance at different rates.
/// When these are mapped to global document IDs, temporary gaps can appear.
/// As a result, simply iterating from 0 to the total document count may encounter invalid IDs near the end.
/// The Document Iterator abstracts this complexity and reliably returns only valid document IDs.
/// # Parameters
/// - docid=None, take>0: **skip first s document IDs**, then **take next t document IDs** of an index.
/// - docid=None, take<0: **skip last s document IDs**, then **take previous t document IDs** of an index.
/// - docid=Some, take>0: **skip next s document IDs**, then **take next t document IDs** of an index, relative to a given document ID, with end-of-index indicator.
/// - docid=Some, take<0: **skip previous s document IDs**, then **take previous t document IDs**, relative to a given document ID, with start-of-index indicator.
/// - take=0: does not make sense, that defies the purpose of get_iterator.
/// - The sign of take indicates the direction of iteration: positive take for forward iteration, negative take for backward iteration.
/// - The skip parameter is always positive, indicating the number of document IDs to skip before taking document IDs. The skip direction is determined by the sign of take too.
/// - include_document: if true, the documents are also retrieved along with their document IDs.
/// Next page:     take last  docid from previous result set, skip=1, take=+page_size
/// Previous page: take first docid from previous result set, skip=1, take=-page_size
/// Returns an IteratorResult, consisting of the number of actually skipped document IDs, and a list of taken document IDs and documents, sorted ascending).
/// Detect end/begin of index during iteration: if returned vec.len() < requested take || if returned skip <requested skip
#[utoipa::path(
    get,
    tag = "Iterator",
    path = "/api/v1/index/{index_id}/doc_id",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),

  params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id", example=0),
        ("document_id" = u64, Query,  description = "document id"),
        ("skip" = u64, Query,  description = "skip document IDs", minimum = 0, example=0),
        ("take" = u64, Query,  description = "take document IDs",  example=-1),
        ("include_deleted" = bool, Query,  description = "include deleted document IDs in results", example=false),
        ("include_document" = bool, Query,  description = "include documents in results", example=false),
        ("fields" = Vec<String>, Query,  description = "fields to include in document. If not specified, all fields are included", example=json!(["title","body"]) ),
    ),
    responses(
        (status = 200, description = "Document ID found, returning an IteratorResult", body = IteratorResult),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn get_iterator_api_get(
    index_arc: &IndexArc,
    document_id: Option<u64>,
    skip: usize,
    take: isize,
    include_deleted: bool,
    include_document: bool,
    fields: Vec<String>,
) -> IteratorResult {
    index_arc
        .get_iterator(
            document_id,
            skip,
            take,
            include_deleted,
            include_document,
            fields,
        )
        .await
}

/// Document iterator
///
/// Document iterator via GET and POST are identical, only the way parameters are passed differ.
/// The document iterator allows to iterate over all document IDs and documents in the entire index, forward or backward.
/// It enables efficient sequential access to every document, even in very large indexes, without running a search.
/// Paging through the index works without collecting document IDs to Min-heap in size-limited RAM first.
/// The iterator guarantees that only valid document IDs are returned, even though document IDs are not strictly continuous.
/// Document IDs can also be fetched in batches, reducing round trips and significantly improving performance, especially when using the REST API.
/// Typical use cases include index export, conversion, analytics, audits, and inspection.
/// Explanation of "eventually continuous" docid:
/// In SeekStorm, document IDs become continuous over time. In a multi-sharded index, each shard maintains its own document ID space.
/// Because documents are distributed across shards in a non-deterministic, load-dependent way, shard-local document IDs advance at different rates.
/// When these are mapped to global document IDs, temporary gaps can appear.
/// As a result, simply iterating from 0 to the total document count may encounter invalid IDs near the end.
/// The Document Iterator abstracts this complexity and reliably returns only valid document IDs.
/// # Parameters
/// - docid=None, take>0: **skip first s document IDs**, then **take next t document IDs** of an index.
/// - docid=None, take<0: **skip last s document IDs**, then **take previous t document IDs** of an index.
/// - docid=Some, take>0: **skip next s document IDs**, then **take next t document IDs** of an index, relative to a given document ID, with end-of-index indicator.
/// - docid=Some, take<0: **skip previous s document IDs**, then **take previous t document IDs**, relative to a given document ID, with start-of-index indicator.
/// - take=0: does not make sense, that defies the purpose of get_iterator.
/// - The sign of take indicates the direction of iteration: positive take for forward iteration, negative take for backward iteration.
/// - The skip parameter is always positive, indicating the number of document IDs to skip before taking document IDs. The skip direction is determined by the sign of take too.
/// - include_document: if true, the documents are also retrieved along with their document IDs.
/// Next page:     take last  docid from previous result set, skip=1, take=+page_size
/// Previous page: take first docid from previous result set, skip=1, take=-page_size
/// Returns an IteratorResult, consisting of the number of actually skipped document IDs, and a list of taken document IDs and documents, sorted ascending).
/// Detect end/begin of index during iteration: if returned vec.len() < requested take || if returned skip <requested skip
#[utoipa::path(
    post,
    tag = "Iterator",
    path = "/api/v1/index/{index_id}/doc_id",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),
    request_body(content = GetIteratorRequest, example=json!({
        "document_id": null,
        "skip": 0,
        "take": -1,
    })),
    responses(
        (status = 200, description = "Document ID found, returning an IteratorResult", body = IteratorResult),
        (status = BAD_REQUEST, description = "index_id invalid or missing"),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn get_iterator_api_post(
    index_arc: &IndexArc,
    document_id: Option<u64>,
    skip: usize,
    take: isize,
    include_deleted: bool,
    include_document: bool,
    fields: Vec<String>,
) -> IteratorResult {
    index_arc
        .get_iterator(
            document_id,
            skip,
            take,
            include_deleted,
            include_document,
            fields,
        )
        .await
}

pub(crate) async fn clear_index_api(index_arc: &IndexArc) -> Result<u64, String> {
    let mut index_mut = index_arc.write().await;
    index_mut.clear_index().await;
    Ok(index_mut.indexed_doc_count().await as u64)
}

/// Query Index
///
/// Query results from index with index_id
/// The following parameters are supported:
/// - Result type
/// - Result sorting
/// - Realtime search
/// - Field filter
/// - Fields to include in search results
/// - Distance fields: derived fields from distance calculations
/// - Highlights: keyword-in-context snippets and term highlighting
/// - Query facets: which facets fields to calculate and return at query time
/// - Facet filter: filter facets by field and value
/// - Result sort: sort results by field and direction
/// - Query type default: default query type, if not specified in query
#[utoipa::path(
    post,
    tag = "Query",
    path = "/api/v1/index/{index_id}/query",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id"),
    ),


    request_body(
        content = SearchRequestObject,
        examples(
            ("Example: Lexical search" = (value = json!(  {
        "query": "detroit",
        "query_vector": null,
        "enable_empty_query": false,
        "offset": 0,
        "length": 10,
        "result_type": "TopkCount",
        "realtime": true,
        "highlights": [
            {
                "field": "",
                "name": "",
                "fragment_number": 0,
                "fragment_size": 0,
                "highlight_markup": true,
                "pre_tags": "",
                "post_tags": ""
            }
        ],
        "field_filter": ["title"],
        "fields": [],
        "query_type_default": "Intersection",
        "query_rewriting": "SearchOnly",
        "search_mode": "Lexical"
    }   ))),
	("Example: Hybrid search" = (value = json!(  {
        "query": "detroit",
        "query_vector": null,
        "enable_empty_query": false,
        "offset": 0,
        "length": 10,
        "result_type": "TopkCount",
        "realtime": true,
        "highlights": [
            {
                "field": "",
                "name": "",
                "fragment_number": 0,
                "fragment_size": 0,
                "highlight_markup": true,
                "pre_tags": "",
                "post_tags": ""
            }
        ],
        "field_filter": ["title"],
        "fields": [],
        "query_type_default": "Intersection",
        "query_rewriting": "SearchOnly",
        "search_mode": {
            "Hybrid": {
                "similarity_threshold": 0.7,
                "ann_mode": {
                    "Nprobe": 55
                }
            }
        }
    }  ))),
("Example: Vector search" = (value = json!(  {
    "query":"",
    "query_vector": [
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013,
        0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026,
        0.027, 0.028, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039,
        0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.051, 0.052,
        0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.064, 0.065,
        0.066, 0.067, 0.068, 0.069, 0.070, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078,
        0.079, 0.080, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.090, 0.091,
        0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.100, 0.101, 0.102, 0.103, 0.104,
        0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117,
        0.118, 0.119, 0.120, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128
    ],
    "offset":0,
    "length":10,
    "result_type": "Topk",
    "realtime": false,
    "search_mode": {
        "Vector":{
            "similarity_threshold": 0.7, 
            "ann_mode": {"Nprobe":55}
        }
    }

}  )))
        )
    ),
    responses(
        (status = 200, description = "Results found, returns the SearchResultObject", body = SearchResultObject),
        (status = BAD_REQUEST, description = "Request object incorrect"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn query_index_api_post(
    index_arc: &IndexArc,
    search_request: SearchRequestObject,
) -> SearchResultObject {
    query_index_api(index_arc, search_request).await
}

/// Query Index
///
/// Query results from index with index_id.
/// Query index via GET is a convenience function, that **offers only a limited set of parameters compared to Query Index via POST**.
/// Always use Query Index via POST for the full set of parameters and maximum flexibility.
/// Query Index via GET is provided for simple queries and quick testing, and to be easily callable from browser address bar, but it is not intended for production use.
#[utoipa::path(
    get,
    tag = "Query",
    path = "/api/v1/index/{index_id}/query",
    params(
        ("apikey" = String, Header, description = "YOUR_SECRET_API_KEY",example="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("index_id" = u64, Path, description = "index id", example=0),
        ("query" = String, Query,  description = "query string", example="hello"),
        ("offset" = u64, Query,  description = "result offset", minimum = 0, example=0),
        ("length" = u64, Query,  description = "result length", minimum = 1, example=10),
        ("realtime" = bool, Query,  description = "include uncommitted documents", example=false),
        ("enable_empty_query" = bool, Query,  description = "allow empty query", example=false)
    ),
    responses(
        (status = 200, description = "Results found, returns the SearchResultObject", body = SearchResultObject),
        (status = BAD_REQUEST, description = "No query specified"),
        (status = NOT_FOUND, description = "Index id does not exist"),
        (status = NOT_FOUND, description = "API key does not exist"),
        (status = UNAUTHORIZED, description = "api_key does not exists"),
        (status = UNAUTHORIZED, description = "api_key missing"),
    )
)]
pub(crate) async fn query_index_api_get(
    index_arc: &IndexArc,
    search_request: SearchRequestObject,
) -> SearchResultObject {
    query_index_api(index_arc, search_request).await
}

use seekstorm::vector::{embedding_from_bytes_be, embedding_from_json};

pub(crate) async fn query_index_api(
    index_arc: &IndexArc,
    search_request: SearchRequestObject,
) -> SearchResultObject {
    let start_time = Instant::now();

    let query_vector = if let Some(value) = search_request.query_vector
        && search_request.search_mode != SearchMode::Lexical
    {
        match &value {
            Value::String(string_base64) => {
                if let Ok(bytes) = decode_bytes_from_base64_string(string_base64)
                    && let Some(embedding) = embedding_from_bytes_be(
                        &bytes,
                        index_arc.read().await.vector_precision,
                        index_arc.read().await.vector_dimensions_original,
                        *IS_SYSTEM_LE,
                    )
                {
                    Some(embedding)
                } else {
                    None
                }
            }
            Value::Array(_) => embedding_from_json(
                &value,
                index_arc.read().await.vector_precision,
                index_arc.read().await.vector_dimensions_original,
            ),
            _ => None,
        }
    } else {
        None
    };

    let result_object = index_arc
        .search(
            search_request.query_string.to_owned(),
            query_vector,
            search_request.query_type_default,
            search_request.search_mode,
            search_request.enable_empty_query,
            search_request.offset,
            search_request.length,
            search_request.result_type,
            search_request.realtime,
            search_request.field_filter,
            search_request.query_facets,
            search_request.facet_filter,
            search_request.result_sort,
            search_request.query_rewriting,
        )
        .await;

    let elapsed_time = start_time.elapsed().as_nanos();

    let return_fields_filter = HashSet::from_iter(search_request.fields);

    let mut results: Vec<Document> = Vec::new();

    if !index_arc.read().await.stored_field_names.is_empty() {
        let highlighter_option = if search_request.highlights.is_empty() {
            None
        } else {
            Some(
                highlighter(
                    index_arc,
                    search_request.highlights,
                    result_object.query_terms.clone(),
                )
                .await,
            )
        };

        for result in result_object.results.iter() {
            match index_arc
                .read()
                .await
                .get_document(
                    result.doc_id,
                    search_request.realtime,
                    &highlighter_option,
                    &return_fields_filter,
                    &search_request.distance_fields,
                )
                .await
            {
                Ok(doc) => {
                    let mut doc = doc;
                    doc.insert("_id".to_string(), result.doc_id.into());
                    doc.insert("_score".to_string(), result.score.into());

                    results.push(doc);
                }
                Err(_e) => {}
            }
        }
    }

    SearchResultObject {
        original_query: result_object.original_query.to_owned(),
        query: result_object.query.to_owned(),
        time: elapsed_time,
        offset: search_request.offset,
        length: search_request.length,
        count: result_object.results.len(),
        count_total: result_object.result_count_total,
        query_terms: result_object.query_terms,
        results,
        facets: result_object.facets,
        suggestions: result_object.suggestions,
    }
}

#[derive(OpenApi, Default)]
#[openapi(paths(
    live_api,
    create_apikey_api,
    get_apikey_indices_info_api,
    delete_apikey_api,
    create_index_api,
    get_index_info_api,
    commit_index_api,
    delete_index_api,
    get_iterator_api_post,
    get_iterator_api_get,
    index_document_api,
    update_document_api,
    index_file_api,
    get_document_api,
    get_file_api,
    delete_document_by_parameter_api,
    delete_document_by_object_api,
    query_index_api_post,
    query_index_api_get,
),
tags(
    (name="Info", description="Return info about the server"),
    (name="API Key", description="Create and delete API keys"),
    (name="Index", description="Create and delete indices"),
    (name="Iterator", description="Iterate through document IDs and documents"),
    (name="Document", description="Index, update, get and delete documents"),
    (name="PDF File", description="Index, and get PDF file"),
    (name="Query", description="Query an index"),
)
)]
#[openapi(info(title = "SeekStorm REST API documentation"))]
#[openapi(servers((url = "http://127.0.0.1", description = "Local SeekStorm server")))]
struct ApiDoc;

pub fn generate_openapi() {
    let openapi = ApiDoc::openapi();

    println!("{}", openapi.to_pretty_json().unwrap());

    let mut path = current_exe().unwrap();
    path.pop();
    let path_json = path.join("openapi.json");
    let path_yml = path.join("openapi.yml");

    serde_json::to_writer_pretty(&File::create(path_json.clone()).unwrap(), &openapi).unwrap();
    fs::write(path_yml.clone(), openapi.to_yaml().unwrap()).unwrap();

    println!(
        "OpenAPI documents generated: {} {}",
        path_json.to_string_lossy(),
        path_yml.to_string_lossy()
    );
}
