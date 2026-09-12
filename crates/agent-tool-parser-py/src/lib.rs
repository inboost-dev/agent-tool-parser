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
fn strip_thinking(text: &str) -> String {
    cleaners::strip_thinking(text)
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

#[pymodule]
fn _accelerated(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ToolError", _py.get_type_bound::<ToolError>())?;
    m.add_class::<PyToolCall>()?;
    m.add_class::<PyToolParser>()?;
    m.add_function(wrap_pyfunction!(parse_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(try_parse_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(try_parse_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(safe_json_loads, m)?)?;
    m.add_function(wrap_pyfunction!(clean_json_str, m)?)?;
    m.add_function(wrap_pyfunction!(strip_thinking, m)?)?;
    Ok(())
}
