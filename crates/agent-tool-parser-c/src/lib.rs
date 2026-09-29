// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

#![allow(clippy::missing_safety_doc)]

use agent_tool_parser_core::{cleaners::clean_json_str, parse_tool_call, parse_tool_calls};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct ATPToolCall {
    pub name: *mut c_char,
    pub args_json: *mut c_char,
    pub raw_source: *mut c_char,
}

#[repr(C)]
pub struct ATPToolCallList {
    pub calls: *mut ATPToolCall,
    pub count: usize,
}

#[repr(C)]
pub struct ATPRawToolCall {
    pub name: *mut c_char,
    pub raw_args: *mut c_char,
    pub raw_source: *mut c_char,
    pub call_id: *mut c_char,
}

#[repr(C)]
pub struct ATPRawToolCallList {
    pub calls: *mut ATPRawToolCall,
    pub count: usize,
}

#[no_mangle]
pub unsafe extern "C" fn atp_parse_tool_call(text: *const c_char) -> *mut ATPToolCall {
    if text.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match parse_tool_call(c_str) {
        Ok(tc) => {
            let args_json = tc.to_canonical_json();
            let name_c = CString::new(tc.name).unwrap_or_default().into_raw();
            let args_c = CString::new(args_json).unwrap_or_default().into_raw();
            let raw_c = CString::new(tc.raw_source).unwrap_or_default().into_raw();

            let boxed = Box::new(ATPToolCall {
                name: name_c,
                args_json: args_c,
                raw_source: raw_c,
            });
            Box::into_raw(boxed)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_parse_tool_calls(text: *const c_char) -> ATPToolCallList {
    if text.is_null() {
        return ATPToolCallList {
            calls: std::ptr::null_mut(),
            count: 0,
        };
    }

    let c_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            return ATPToolCallList {
                calls: std::ptr::null_mut(),
                count: 0,
            }
        }
    };

    match parse_tool_calls(c_str) {
        Ok(tcs) => {
            let mut raw_calls: Vec<ATPToolCall> = Vec::with_capacity(tcs.len());
            for tc in tcs {
                let args_json = tc.to_canonical_json();
                let name_c = CString::new(tc.name).unwrap_or_default().into_raw();
                let args_c = CString::new(args_json).unwrap_or_default().into_raw();
                let raw_c = CString::new(tc.raw_source).unwrap_or_default().into_raw();

                raw_calls.push(ATPToolCall {
                    name: name_c,
                    args_json: args_c,
                    raw_source: raw_c,
                });
            }
            let count = raw_calls.len();
            let mut boxed_slice = raw_calls.into_boxed_slice();
            let ptr = boxed_slice.as_mut_ptr();
            std::mem::forget(boxed_slice);

            ATPToolCallList { calls: ptr, count }
        }
        Err(_) => ATPToolCallList {
            calls: std::ptr::null_mut(),
            count: 0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_free_tool_call(call: *mut ATPToolCall) {
    if call.is_null() {
        return;
    }
    let call = Box::from_raw(call);
    if !call.name.is_null() {
        let _ = CString::from_raw(call.name);
    }
    if !call.args_json.is_null() {
        let _ = CString::from_raw(call.args_json);
    }
    if !call.raw_source.is_null() {
        let _ = CString::from_raw(call.raw_source);
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_free_tool_call_list(list: ATPToolCallList) {
    if list.calls.is_null() || list.count == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(list.calls, list.count);
    for item in slice.iter_mut() {
        if !item.name.is_null() {
            let _ = CString::from_raw(item.name);
        }
        if !item.args_json.is_null() {
            let _ = CString::from_raw(item.args_json);
        }
        if !item.raw_source.is_null() {
            let _ = CString::from_raw(item.raw_source);
        }
    }
    let _ = Box::from_raw(list.calls);
}

#[no_mangle]
pub unsafe extern "C" fn atp_extract_raw_tool_calls(text: *const c_char) -> ATPRawToolCallList {
    if text.is_null() {
        return ATPRawToolCallList {
            calls: std::ptr::null_mut(),
            count: 0,
        };
    }

    let c_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            return ATPRawToolCallList {
                calls: std::ptr::null_mut(),
                count: 0,
            }
        }
    };

    let raws = agent_tool_parser_core::extract_raw_tool_calls(c_str);
    let mut raw_calls: Vec<ATPRawToolCall> = Vec::with_capacity(raws.len());
    for r in raws {
        let name_c = CString::new(r.name).unwrap_or_default().into_raw();
        let args_c = CString::new(r.raw_args).unwrap_or_default().into_raw();
        let raw_c = CString::new(r.raw_source).unwrap_or_default().into_raw();
        let id_c = r
            .call_id
            .map(|id| CString::new(id).unwrap_or_default().into_raw())
            .unwrap_or(std::ptr::null_mut());

        raw_calls.push(ATPRawToolCall {
            name: name_c,
            raw_args: args_c,
            raw_source: raw_c,
            call_id: id_c,
        });
    }
    let count = raw_calls.len();
    let mut boxed_slice = raw_calls.into_boxed_slice();
    let ptr = boxed_slice.as_mut_ptr();
    std::mem::forget(boxed_slice);

    ATPRawToolCallList { calls: ptr, count }
}

#[no_mangle]
pub unsafe extern "C" fn atp_raw_parse_args(call: *const ATPRawToolCall) -> *mut c_char {
    if call.is_null() || (*call).name.is_null() || (*call).raw_args.is_null() {
        return std::ptr::null_mut();
    }

    let name = match CStr::from_ptr((*call).name).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let raw_args = match CStr::from_ptr((*call).raw_args).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let raw_source = if (*call).raw_source.is_null() {
        ""
    } else {
        CStr::from_ptr((*call).raw_source).to_str().unwrap_or("")
    };

    let raw = agent_tool_parser_core::RawToolCall::new(name, raw_args, raw_source);
    match raw.parse_args() {
        Ok(val) => {
            let json_str = serde_json::to_string(&val).unwrap_or_default();
            CString::new(json_str).unwrap_or_default().into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_free_raw_tool_call(call: *mut ATPRawToolCall) {
    if call.is_null() {
        return;
    }
    let call = Box::from_raw(call);
    if !call.name.is_null() {
        let _ = CString::from_raw(call.name);
    }
    if !call.raw_args.is_null() {
        let _ = CString::from_raw(call.raw_args);
    }
    if !call.raw_source.is_null() {
        let _ = CString::from_raw(call.raw_source);
    }
    if !call.call_id.is_null() {
        let _ = CString::from_raw(call.call_id);
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_free_raw_tool_call_list(list: ATPRawToolCallList) {
    if list.calls.is_null() || list.count == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(list.calls, list.count);
    for item in slice.iter_mut() {
        if !item.name.is_null() {
            let _ = CString::from_raw(item.name);
        }
        if !item.raw_args.is_null() {
            let _ = CString::from_raw(item.raw_args);
        }
        if !item.raw_source.is_null() {
            let _ = CString::from_raw(item.raw_source);
        }
        if !item.call_id.is_null() {
            let _ = CString::from_raw(item.call_id);
        }
    }
    let _ = Box::from_raw(list.calls);
}

#[no_mangle]
pub unsafe extern "C" fn atp_clean_json_str(json_str: *const c_char) -> *mut c_char {
    if json_str.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = match CStr::from_ptr(json_str).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let cleaned = clean_json_str(c_str);
    CString::new(cleaned).unwrap_or_default().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn atp_canonicalize_json(json_str: *const c_char) -> *mut c_char {
    if json_str.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = match CStr::from_ptr(json_str).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let canon = agent_tool_parser_core::canonicalize_arguments_string(c_str);
    CString::new(canon).unwrap_or_default().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn atp_repair_unescaped_quotes(json_str: *const c_char) -> *mut c_char {
    if json_str.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = match CStr::from_ptr(json_str).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let repaired = agent_tool_parser_core::repair_unescaped_quotes(c_str);
    CString::new(repaired).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn atp_detect_vector_engine() -> *const c_char {
    match agent_tool_parser_core::detect_vector_engine() {
        agent_tool_parser_core::CpuVectorEngine::Avx2_256 => c"avx2_256".as_ptr(),
        agent_tool_parser_core::CpuVectorEngine::Sse42_128 => c"sse42_128".as_ptr(),
        agent_tool_parser_core::CpuVectorEngine::Neon_128 => c"neon_128".as_ptr(),
        agent_tool_parser_core::CpuVectorEngine::Scalar_Fallback => c"scalar_fallback".as_ptr(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn atp_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_abi_parse_and_free() {
        unsafe {
            let input = CString::new(
                r#"<invoke name="bash"><parameter name="cmd">pwd</parameter></invoke>"#,
            )
            .unwrap();
            let call_ptr = atp_parse_tool_call(input.as_ptr());
            assert!(!call_ptr.is_null());

            let name = CStr::from_ptr((*call_ptr).name).to_str().unwrap();
            assert_eq!(name, "bash");

            let args = CStr::from_ptr((*call_ptr).args_json).to_str().unwrap();
            assert!(args.contains("pwd"));

            atp_free_tool_call(call_ptr);
        }
    }

    #[test]
    fn test_c_abi_clean_json() {
        unsafe {
            let input = CString::new("{'foo': 'bar'}").unwrap();
            let cleaned_ptr = atp_clean_json_str(input.as_ptr());
            assert!(!cleaned_ptr.is_null());

            let cleaned = CStr::from_ptr(cleaned_ptr).to_str().unwrap();
            assert_eq!(cleaned, r#"{"foo": "bar"}"#);

            atp_free_string(cleaned_ptr);
        }
    }

    #[test]
    fn test_c_abi_canonicalize_json() {
        unsafe {
            let input = CString::new(r#"{"z": 1, "a": 2}"#).unwrap();
            let canon_ptr = atp_canonicalize_json(input.as_ptr());
            assert!(!canon_ptr.is_null());

            let canon = CStr::from_ptr(canon_ptr).to_str().unwrap();
            assert_eq!(canon, r#"{"a":2,"z":1}"#);

            atp_free_string(canon_ptr);
        }
    }

    #[test]
    fn test_c_abi_extract_raw() {
        unsafe {
            let input = CString::new(
                r#"```json
{"name": "fetch", "arguments": {"id": 10}}
```"#,
            )
            .unwrap();
            let raw_list = atp_extract_raw_tool_calls(input.as_ptr());
            assert_eq!(raw_list.count, 1);
            assert!(!raw_list.calls.is_null());

            let first = &*raw_list.calls;
            let name = CStr::from_ptr(first.name).to_str().unwrap();
            assert_eq!(name, "fetch");

            let parsed_args_ptr = atp_raw_parse_args(first);
            assert!(!parsed_args_ptr.is_null());
            let parsed_json = CStr::from_ptr(parsed_args_ptr).to_str().unwrap();
            assert!(parsed_json.contains("10"));

            atp_free_string(parsed_args_ptr);
            atp_free_raw_tool_call_list(raw_list);
        }
    }

    #[test]
    fn test_c_abi_repair_unescaped_quotes() {
        unsafe {
            let input = CString::new(r#"{"cmd": "echo "hello" >> log.txt"}"#).unwrap();
            let repaired_ptr = atp_repair_unescaped_quotes(input.as_ptr());
            assert!(!repaired_ptr.is_null());
            let repaired = CStr::from_ptr(repaired_ptr).to_str().unwrap();
            assert_eq!(repaired, r#"{"cmd": "echo \"hello\" >> log.txt"}"#);
            atp_free_string(repaired_ptr);
        }
    }

    #[test]
    fn test_c_abi_detect_vector_engine() {
        let engine_ptr = atp_detect_vector_engine();
        assert!(!engine_ptr.is_null());
        unsafe {
            let s = CStr::from_ptr(engine_ptr).to_str().unwrap();
            assert!(!s.is_empty());
        }
    }
}
