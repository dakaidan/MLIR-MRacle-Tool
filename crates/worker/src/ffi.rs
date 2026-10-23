use mlir_mracle::protocol::Request;
use std::ffi::{CStr, CString, c_char};

/// Campaign protocol handed to the engine; layout must match `mracle.h`.
#[repr(C)]
struct CampaignOptions {
    comparison: u32,
    campaigns_initial: i32,
    campaigns_max: i32,
    executions_per_campaign: i32,
    team_size: i32,
    bootstrap_replicates: i32,
    tvd_threshold: f64,
    min_change: f64,
    confidence_level: f64,
}

impl From<&Request> for CampaignOptions {
    fn from(request: &Request) -> Self {
        Self {
            comparison: request.comparison as u32,
            campaigns_initial: request.campaigns,
            campaigns_max: request.max_campaigns,
            executions_per_campaign: request.executions_per_campaign,
            team_size: request.team_size,
            bootstrap_replicates: request.bootstrap_replicates,
            tvd_threshold: request.tvd_threshold,
            min_change: request.min_change,
            confidence_level: request.confidence_level,
        }
    }
}

unsafe extern "C" {
    #[cfg(mracle_tsan)]
    fn __tsan_init();
    fn mracle_abi_version() -> u32;
    fn mracle_run(
        input: *const c_char,
        output: *const c_char,
        transform: *const c_char,
        model: *const c_char,
        seed: u32,
        reps: i32,
        replay_reps: i32,
        max_runs: i32,
        max_apply: i32,
        backend: u32,
        tsan_triage: u32,
        campaign: *const CampaignOptions,
        result: *mut *mut c_char,
    ) -> i32;
    fn mracle_free(result: *mut c_char);
}
/// Only called once in a dedicated worker. Foreign memory never crosses allocators.
pub fn run(request: &Request) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let input = CString::new(request.input.to_str().ok_or("non-UTF8 input path")?)?;
    let output = CString::new(request.artifacts.to_str().ok_or("non-UTF8 artifact path")?)?;
    let transform = CString::new(request.transform.as_str())?;
    let model = CString::new(request.model.as_str())?;
    let campaign = CampaignOptions::from(request);
    let mut ptr = std::ptr::null_mut();
    // SAFETY: all C strings live through the call, result points to writable storage;
    // the ABI owns the returned NUL-terminated allocation until mracle_free.
    let (code, bytes) = unsafe {
        // Idempotent initialization also gives Rust a direct runtime reference,
        // retaining TSan before libc even under --as-needed static linking.
        #[cfg(mracle_tsan)]
        __tsan_init();
        if mracle_abi_version() != 3 {
            return Err("unsupported native ABI".into());
        }
        let code = mracle_run(
            input.as_ptr(),
            output.as_ptr(),
            transform.as_ptr(),
            model.as_ptr(),
            request.seed,
            request.reps,
            request.replay_reps,
            request.max_runs,
            request.max_apply,
            request.backend as u32,
            u32::from(request.tsan_triage),
            &campaign,
            &mut ptr,
        );
        let bytes = if ptr.is_null() {
            Vec::new()
        } else {
            CStr::from_ptr(ptr).to_bytes().to_vec()
        };
        mracle_free(ptr);
        (code, bytes)
    };
    if code != 0 {
        return Err(format!("native error {code}: {}", String::from_utf8_lossy(&bytes)).into());
    }
    Ok(bytes)
}
