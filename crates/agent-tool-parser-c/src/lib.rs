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
            let name_c = CString::new(tc.name).unwrap_or_default().into_raw();
            let args_json = serde_json::to_string(&tc.args).unwrap_or_default();
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
                let name_c = CString::new(tc.name).unwrap_or_default().into_raw();
                let args_json = serde_json::to_string(&tc.args).unwrap_or_default();
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
}
