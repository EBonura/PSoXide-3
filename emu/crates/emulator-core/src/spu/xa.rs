/// Per-channel decoder history for XA ADPCM blocks. The filter
/// uses the last two decoded samples (`y0` = most recent,
/// `y1` = second-most-recent) as feedback. Callers hold one of
/// these per stereo channel (or one total for mono).
#[derive(Default, Clone, Debug)]
pub struct XaDecoderState {
    y0: i32,
    y1: i32,
}

impl XaDecoderState {
    /// Fresh decoder history — silence as prev samples.
    pub fn new() -> Self {
        Self { y0: 0, y1: 0 }
    }

    /// Reset history to silence between XA files.
    pub fn reset(&mut self) {
        self.y0 = 0;
        self.y1 = 0;
    }
}

/// XA ADPCM filter coefficients `k0, k1` in Q10 form. Four filter
/// IDs match the real-hardware decode table. Pattern matches
/// Redux's `decode_xa.cc::s_K0/s_K1` at `(1<<SHC = 1024)`.
const XA_FILTER: [(i32, i32); 4] = [(0, 0), (960, 0), (1840, -832), (1568, -880)];

/// Decode 28 ADPCM samples (one "sound unit") from an XA block.
/// - `filter_range` — packed byte: high nibble = filter ID (0..=3,
///   values >3 are reserved), low nibble = range (output shift).
/// - `data` — seven 16-bit packed words, laid out exactly like
///   Redux's `decode_xa.cc` before it calls `ADPCM_DecodeBlock16`.
///   Each word carries four 4-bit samples.
/// - `state` — in/out filter history; mutates across calls within a
///   sound group.
///
/// Writes 28 output samples into `out[0], out[stride], out[2*stride], ...`.
/// Stride = 2 for interleaved stereo, 1 for mono.
pub fn xa_decode_block(
    state: &mut XaDecoderState,
    filter_range: u8,
    data: &[u16],
    out: &mut [i16],
    stride: usize,
) {
    let filter_id = ((filter_range >> 4) & 0x0F).min(3) as usize;
    let range = (filter_range & 0x0F) as u32;
    let (k0, k1) = XA_FILTER[filter_id];
    let mut y0 = state.y0;
    let mut y1 = state.y1;

    // Match Redux's `ADPCM_DecodeBlock16` exactly: unpack one packed
    // 16-bit word into x0..x3 (high nibble first), run the IIR filter,
    // clamp in Q4, then emit 16-bit PCM.
    for (i, &word) in data.iter().take(7).enumerate() {
        let expand = |shift: u32| -> i32 {
            let nib = ((((word as u32) << shift) & 0xF000) as u16) as i16 as i32;
            (nib >> range) << 4
        };

        let mut x3 = expand(0);
        let mut x2 = expand(4);
        let mut x1 = expand(8);
        let mut x0 = expand(12);

        x0 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x0;
        x1 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x1;
        x2 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x2;
        x3 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x3;

        let decoded = [x0, x1, x2, x3];
        for (n, &sample) in decoded.iter().enumerate() {
            let clamped = sample.clamp(-32768 << 4, 32767 << 4);
            let idx = (i * 4 + n) * stride;
            if idx < out.len() {
                out[idx] = (clamped >> 4) as i16;
            }
        }
    }
    state.y0 = y0;
    state.y1 = y1;
}
