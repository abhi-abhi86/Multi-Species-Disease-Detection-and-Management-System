# TODO: Enable GPU Training for Disease Classifier

## Completed Tasks
- [x] Modify device selection in train_disease_classifier.py to prioritize MPS (macOS GPU), then CUDA, then CPU.

## Next Steps
- [ ] Test the training script to verify GPU usage (run `python train_disease_classifier.py` and check if it prints "MPS" or "CUDA").
- [ ] If MPS is not available, ensure PyTorch version supports MPS (requires PyTorch 1.12+ on macOS with Apple Silicon).
- [ ] Monitor training performance for speed improvements.
