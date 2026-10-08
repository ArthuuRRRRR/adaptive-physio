# Adaptive Physio

Adaptive Coarse-to-Fine Multimodal Physiological Generation

## Objective

Generate variable-resolution physiological recordings while
preserving task-relevant information and reducing computational cost.

## Modalities

- EEG
- ECG
- Respiration

## Architecture

1. Physiological Register Encoder
2. Hierarchical FSQ Tokenizer (16 / 64 / 128 tokens)
3. Conditional Flow Decoder
4. Adaptive Agent (planned)
5. Variable-Resolution Night Generator (planned)

## Current Status

- Register Encoder implemented and tested
- FSQ Quantizer implemented and tested
- Gaussian granularity targets implemented
- Conditional Flow Decoder prototype implemented
- SHHS1 EDF DataLoader implemented
- Training and evaluation in progress
