// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

#![allow(unexpected_cfgs)]
#![allow(clippy::useless_conversion)]

use agent_tool_parser_core::{
    cleaners, parse_tool_call as core_parse_call, parse_tool_calls as core_parse_calls,
    ToolParser as CoreToolParser, ToolParserConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;
use std::collections::{HashMap, HashSet};

pyo3::create_exception!(_accelerated, ToolError, pyo3::exceptions::PyValueError);

#[pyclass(name = "ToolCall", module = "_accelerated")]
#[derive(Clone)]
pub struct PyToolCall {
    #[pyo3(get)]
    pub name: String,
    pub args_val: serde_json::Value,
    #[pyo3(get)]
    pub raw_source: String,
}

#[pymethods]
impl PyToolCall {
    #[new]
    #[pyo3(signature = (name, args=None, raw_source=String::new()))]
    fn new<'py>(
        _py: Python<'py>,
        name: String,
        args: Option<Bound<'py, PyAny>>,
        raw_source: String,
    ) -> PyResult<Self> {
        let args_val = if let Some(a) = args {
            pythonize::depythonize(&a)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        Ok(Self {
            name,
            args_val,
            raw_source,
        })
    }

    #[getter]
    fn args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.args_val).map_err(|e| ToolError::new_err(e.to_string()))
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("name", &self.name)?;
        dict.set_item("args", self.args(py)?)?;
        dict.set_item("raw_source", &self.raw_source)?;
        Ok(dict)
    }

    fn to_canonical_json(&self) -> String {
        agent_tool_parser_core::canonical::canonical_json_dumps(&self.args_val)
    }

    #[pyo3(signature = (call_id=None, canonical=true))]
    fn to_openai_tool_call<'py>(
        &self,
        py: Python<'py>,
        call_id: Option<String>,
        canonical: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let id = call_id.unwrap_or_else(|| format!("call_{}", self.name));
        let arguments = if canonical {
            self.to_canonical_json()
        } else {
            serde_json::to_string(&self.args_val).unwrap_or_default()
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("id", id)?;
        dict.set_item("type", "function")?;

        let func = PyDict::new_bound(py);
        func.set_item("name", &self.name)?;
        func.set_item("arguments", arguments)?;

        dict.set_item("function", func)?;
        Ok(dict)
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let args_repr = self.args(py)?.repr()?.to_string();
        Ok(format!(
            "ToolCall(name={:?}, args={})",
            self.name, args_repr
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.name == other.name
            && self.args_val == other.args_val
            && self.raw_source == other.raw_source
    }
}

impl PyToolCall {
    fn from_core(tc: agent_tool_parser_core::ToolCall) -> Self {
        Self {
            name: tc.name,
            args_val: tc.args,
            raw_source: tc.raw_source,
        }
    }
}

#[pyfunction]
fn parse_tool_call(text: &str) -> PyResult<PyToolCall> {
    core_parse_call(text)
        .map(PyToolCall::from_core)
        .map_err(|e| ToolError::new_err(e.message))
}

#[pyfunction]
fn parse_tool_calls(text: &str) -> PyResult<Vec<PyToolCall>> {
    core_parse_calls(text)
        .map(|list| list.into_iter().map(PyToolCall::from_core).collect())
        .map_err(|e| ToolError::new_err(e.message))
}

#[pyfunction]
fn try_parse_tool_call(text: &str) -> Option<PyToolCall> {
    core_parse_call(text).ok().map(PyToolCall::from_core)
}

#[pyfunction]
fn try_parse_tool_calls(text: &str) -> Vec<PyToolCall> {
    core_parse_calls(text)
        .unwrap_or_default()
        .into_iter()
        .map(PyToolCall::from_core)
        .collect()
}

#[pyfunction]
#[pyo3(signature = (s, default=None))]
fn safe_json_loads<'py>(
    py: Python<'py>,
    s: &str,
    default: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match cleaners::safe_json_loads(s) {
        Some(val) => pythonize(py, &val).map_err(|e| PyValueError::new_err(e.to_string())),
        None => match default {
            Some(d) => Ok(d),
            None => Ok(py.None().into_bound(py)),
        },
    }
}

#[pyfunction]
fn clean_json_str(s: &str) -> String {
    cleaners::clean_json_str(s)
}

#[pyfunction]
fn repair_unescaped_quotes(s: &str) -> String {
    cleaners::repair_unescaped_quotes(s)
}

#[pyfunction]
fn detect_vector_engine() -> &'static str {
    agent_tool_parser_core::detect_vector_engine().as_str()
}

#[pyfunction]
fn normalize_truncated_json_prefix(s: &str) -> String {
    cleaners::normalize_truncated_json_prefix(s).into_owned()
}

#[pyfunction]
fn extract_json_objects(text: &str) -> Vec<String> {
    cleaners::extract_json_objects(text)
}

#[pyfunction]
fn strip_thinking(text: &str) -> String {
    cleaners::strip_thinking(text)
}

#[pyfunction]
fn canonicalize_arguments_string(s: &str) -> String {
    agent_tool_parser_core::canonicalize_arguments_string(s)
}

#[pyfunction]
fn canonical_json_dumps<'py>(_py: Python<'py>, obj: Bound<'py, PyAny>) -> PyResult<String> {
    let val: serde_json::Value =
        pythonize::depythonize(&obj).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(agent_tool_parser_core::canonical_json_dumps(&val))
}

#[pyfunction]
fn canonical_sort_keys<'py>(
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let val: serde_json::Value =
        pythonize::depythonize(&obj).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sorted = agent_tool_parser_core::canonical_sort_keys(&val);
    pythonize(py, &sorted).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "ToolParser", module = "_accelerated")]
pub struct PyToolParser {
    inner: CoreToolParser,
}

#[pymethods]
impl PyToolParser {
    #[new]
    #[pyo3(signature = (
        allowed_tools=None,
        tool_aliases=None,
        param_aliases=None,
        code_param_keys=None,
        strip_thinking=true,
        allow_shell_fallback=false
    ))]
    fn new(
        allowed_tools: Option<HashSet<String>>,
        tool_aliases: Option<HashMap<String, String>>,
        param_aliases: Option<HashMap<String, String>>,
        code_param_keys: Option<HashSet<String>>,
        strip_thinking: bool,
        allow_shell_fallback: bool,
    ) -> Self {
        let mut config = ToolParserConfig::default();
        if let Some(at) = allowed_tools {
            config.allowed_tools = Some(at);
        }
        if let Some(ta) = tool_aliases {
            for (k, v) in ta {
                config
                    .tool_aliases
                    .insert(k.to_lowercase(), v.to_lowercase());
            }
        }
        if let Some(pa) = param_aliases {
            for (k, v) in pa {
                config.param_aliases.insert(k.to_lowercase(), v);
            }
        }
        if let Some(cp) = code_param_keys {
            config.code_param_keys = cp;
        }
        config.strip_thinking = strip_thinking;
        config.allow_shell_fallback = allow_shell_fallback;

        Self {
            inner: CoreToolParser::with_config(config),
        }
    }

    fn parse(&self, text: &str) -> PyResult<PyToolCall> {
        self.inner
            .parse(text)
            .map(PyToolCall::from_core)
            .map_err(|e| ToolError::new_err(e.message))
    }

    fn parse_calls(&self, text: &str) -> PyResult<Vec<PyToolCall>> {
        self.inner
            .parse_calls(text)
            .map(|list| list.into_iter().map(PyToolCall::from_core).collect())
            .map_err(|e| ToolError::new_err(e.message))
    }

    fn parse_all(&self, text: &str) -> PyResult<Vec<PyToolCall>> {
        self.parse_calls(text)
    }

    fn parse_many(&self, text: &str) -> PyResult<Vec<PyToolCall>> {
        self.parse_calls(text)
    }

    fn try_parse(&self, text: &str) -> Option<PyToolCall> {
        self.inner.try_parse(text).map(PyToolCall::from_core)
    }

    fn try_parse_calls(&self, text: &str) -> Vec<PyToolCall> {
        self.inner
            .try_parse_calls(text)
            .into_iter()
            .map(PyToolCall::from_core)
            .collect()
    }

    fn try_parse_all(&self, text: &str) -> Vec<PyToolCall> {
        self.try_parse_calls(text)
    }

    fn try_parse_many(&self, text: &str) -> Vec<PyToolCall> {
        self.try_parse_calls(text)
    }
}

#[pyclass(name = "RawToolCall", module = "_accelerated")]
#[derive(Clone)]
pub struct PyRawToolCall {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub raw_args: String,
    #[pyo3(get)]
    pub raw_source: String,
    #[pyo3(get)]
    pub call_id: Option<String>,
}

#[pymethods]
impl PyRawToolCall {
    #[new]
    #[pyo3(signature = (name, raw_args, raw_source, call_id=None))]
    fn new(name: String, raw_args: String, raw_source: String, call_id: Option<String>) -> Self {
        Self {
            name,
            raw_args,
            raw_source,
            call_id,
        }
    }

    fn parse_args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let raw =
            agent_tool_parser_core::RawToolCall::new(&self.name, &self.raw_args, &self.raw_source);
        let val = raw
            .parse_args()
            .map_err(|e| ToolError::new_err(e.message))?;
        pythonize(py, &val).map_err(|e| ToolError::new_err(e.to_string()))
    }

    fn to_tool_call(&self) -> PyResult<PyToolCall> {
        let raw =
            agent_tool_parser_core::RawToolCall::new(&self.name, &self.raw_args, &self.raw_source);
        let tc = raw
            .to_tool_call()
            .map_err(|e| ToolError::new_err(e.message))?;
        Ok(PyToolCall::from_core(tc))
    }

    fn __repr__(&self) -> String {
        format!(
            "RawToolCall(name={:?}, raw_args={:?})",
            self.name, self.raw_args
        )
    }
}

#[pyfunction]
fn extract_raw_tool_calls(text: &str) -> Vec<PyRawToolCall> {
    agent_tool_parser_core::extract_raw_tool_calls(text)
        .into_iter()
        .map(|r| PyRawToolCall {
            name: r.name,
            raw_args: r.raw_args,
            raw_source: r.raw_source,
            call_id: r.call_id,
        })
        .collect()
}

#[pyfunction]
fn try_extract_raw_tool_call(text: &str) -> Option<PyRawToolCall> {
    extract_raw_tool_calls(text).into_iter().next()
}

#[pyclass(name = "StreamEvent", module = "_accelerated")]
#[derive(Clone)]
pub struct PyStreamEvent {
    #[pyo3(get)]
    pub event_type: String,
    #[pyo3(get)]
    pub text: Option<String>,
    #[pyo3(get)]
    pub thinking: Option<String>,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub call_id: Option<String>,
    #[pyo3(get)]
    pub delta: Option<String>,
    #[pyo3(get)]
    pub tool_call: Option<PyToolCall>,
}

impl PyStreamEvent {
    fn from_core(e: agent_tool_parser_core::StreamEvent) -> Self {
        match e {
            agent_tool_parser_core::StreamEvent::Text(t) => Self {
                event_type: "text".to_string(),
                text: Some(t),
                thinking: None,
                name: None,
                call_id: None,
                delta: None,
                tool_call: None,
            },
            agent_tool_parser_core::StreamEvent::Thinking(t) => Self {
                event_type: "thinking".to_string(),
                text: None,
                thinking: Some(t),
                name: None,
                call_id: None,
                delta: None,
                tool_call: None,
            },
            agent_tool_parser_core::StreamEvent::ToolCallStarted { name, call_id } => Self {
                event_type: "tool_call_started".to_string(),
                text: None,
                thinking: None,
                name: Some(name),
                call_id,
                delta: None,
                tool_call: None,
            },
            agent_tool_parser_core::StreamEvent::ToolCallArgumentsChunk { name, delta } => Self {
                event_type: "tool_call_arguments_chunk".to_string(),
                text: None,
                thinking: None,
                name: Some(name),
                call_id: None,
                delta: Some(delta),
                tool_call: None,
            },
            agent_tool_parser_core::StreamEvent::ToolCallCompleted(tc) => Self {
                event_type: "tool_call_completed".to_string(),
                text: None,
                thinking: None,
                name: None,
                call_id: None,
                delta: None,
                tool_call: Some(PyToolCall::from_core(tc)),
            },
        }
    }
}

#[pymethods]
impl PyStreamEvent {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        event_type,
        text=None,
        thinking=None,
        name=None,
        call_id=None,
        delta=None,
        tool_call=None,
        content=None,
        tool_name=None,
    ))]
    fn new(
        event_type: String,
        mut text: Option<String>,
        mut thinking: Option<String>,
        mut name: Option<String>,
        call_id: Option<String>,
        mut delta: Option<String>,
        tool_call: Option<PyToolCall>,
        content: Option<String>,
        tool_name: Option<String>,
    ) -> Self {
        if let Some(c) = content {
            match event_type.as_str() {
                "thinking" => thinking = Some(c),
                "text" => text = Some(c),
                "tool_call_arguments_chunk" => delta = Some(c),
                _ => {}
            }
        }
        if tool_name.is_some() && name.is_none() {
            name = tool_name;
        }
        Self {
            event_type,
            text,
            thinking,
            name,
            call_id,
            delta,
            tool_call,
        }
    }

    #[getter]
    fn is_text(&self) -> bool {
        self.event_type == "text"
    }

    #[getter]
    fn is_thinking(&self) -> bool {
        self.event_type == "thinking"
    }

    #[getter]
    fn is_tool_call_started(&self) -> bool {
        self.event_type == "tool_call_started"
    }

    #[getter]
    fn is_tool_call_arguments_chunk(&self) -> bool {
        self.event_type == "tool_call_arguments_chunk"
    }

    #[getter]
    fn is_tool_call_completed(&self) -> bool {
        self.event_type == "tool_call_completed"
    }

    #[getter]
    fn is_tool_call(&self) -> bool {
        matches!(
            self.event_type.as_str(),
            "tool_call_started" | "tool_call_arguments_chunk" | "tool_call_completed"
        )
    }

    #[getter]
    fn content(&self) -> String {
        if let Some(ref t) = self.text {
            t.clone()
        } else if let Some(ref th) = self.thinking {
            th.clone()
        } else if let Some(ref d) = self.delta {
            d.clone()
        } else {
            String::new()
        }
    }

    #[getter]
    fn tool_name(&self) -> Option<String> {
        if let Some(ref n) = self.name {
            Some(n.clone())
        } else {
            self.tool_call.as_ref().map(|tc| tc.name.clone())
        }
    }

    fn __repr__(&self) -> String {
        format!("StreamEvent(type={:?})", self.event_type)
    }
}

#[pyclass(name = "StreamingToolParser", module = "_accelerated")]
pub struct PyStreamingToolParser {
    inner: agent_tool_parser_core::StreamingToolParser,
}

#[pymethods]
impl PyStreamingToolParser {
    #[new]
    #[pyo3(signature = (
        allowed_tools=None,
        tool_aliases=None,
        param_aliases=None,
        allow_shell_fallback=false,
    ))]
    fn new(
        allowed_tools: Option<HashSet<String>>,
        tool_aliases: Option<HashMap<String, String>>,
        param_aliases: Option<HashMap<String, String>>,
        allow_shell_fallback: bool,
    ) -> Self {
        let mut config = ToolParserConfig::default();
        if let Some(at) = allowed_tools {
            config.allowed_tools = Some(at);
        }
        if let Some(ta) = tool_aliases {
            config.tool_aliases = ta;
        }
        if let Some(pa) = param_aliases {
            config.param_aliases = pa;
        }
        config.allow_shell_fallback = allow_shell_fallback;

        Self {
            inner: agent_tool_parser_core::StreamingToolParser::with_config(config),
        }
    }

    fn feed(&mut self, chunk: &str) -> Vec<PyStreamEvent> {
        self.inner
            .feed(chunk)
            .into_iter()
            .map(PyStreamEvent::from_core)
            .collect()
    }

    fn finish(&mut self) -> Vec<PyStreamEvent> {
        self.inner
            .finish()
            .into_iter()
            .map(PyStreamEvent::from_core)
            .collect()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pymodule]
fn _accelerated(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ToolError", _py.get_type_bound::<ToolError>())?;
    m.add_class::<PyToolCall>()?;
    m.add_class::<PyToolParser>()?;
    m.add_class::<PyRawToolCall>()?;
    m.add_class::<PyStreamEvent>()?;
    m.add_class::<PyStreamingToolParser>()?;
    m.add_function(wrap_pyfunction!(parse_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(try_parse_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(try_parse_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(extract_raw_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(try_extract_raw_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(safe_json_loads, m)?)?;
    m.add_function(wrap_pyfunction!(clean_json_str, m)?)?;
    m.add_function(wrap_pyfunction!(repair_unescaped_quotes, m)?)?;
    m.add_function(wrap_pyfunction!(detect_vector_engine, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_truncated_json_prefix, m)?)?;

    m.add_function(wrap_pyfunction!(extract_json_objects, m)?)?;
    m.add_function(wrap_pyfunction!(strip_thinking, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_arguments_string, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_json_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_sort_keys, m)?)?;
    Ok(())
}
