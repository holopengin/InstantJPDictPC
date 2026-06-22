// Burn model definitions — generated from ONNX by burn-onnx

// Detection model (always included)
pub mod meiki_text_detect;

// Recognition models (gated behind feature flag due to API mismatches)
#[cfg(feature = "burn-recognition")]
pub mod meiki_text_rec_horizontal;

#[cfg(feature = "burn-recognition")]
pub mod meiki_text_rec_vertical;
