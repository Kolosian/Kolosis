# Contributing to Kolosis

Thank you for your interest in contributing to Kolosis! We welcome contributions of all kinds.

## How to Contribute

### 🐛 Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU, etc.)

### 💡 Suggesting Features

We love new ideas! Open an issue with:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if you have ideas)

### 🖥️ Contributing GPU Resources

**High Priority**: We need help running experiments!

If you have GPU access, you can help by:
1. Running baseline comparison (12 hours on T4/A100)
2. Training Kolosis + Temporal (12 hours on T4/A100)
3. Testing on additional datasets

See [experiments/wikitext/](experiments/wikitext/) for training scripts.

### 📝 Code Contributions

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Add tests** if applicable
5. **Run existing tests**: `python -m pytest tests/`
6. **Commit**: `git commit -m "Add: your feature description"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open a Pull Request**

### Code Style

- Follow PEP 8
- Add docstrings to all functions/classes
- Keep functions focused and modular
- Add comments for complex logic

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test on multiple configurations if possible

## Priority Areas

### 🔥 High Priority
- [ ] Baseline GPT comparison (needs GPU)
- [ ] Temporal attention validation (needs GPU)
- [ ] Additional dataset benchmarks
- [ ] Pre-trained model checkpoints

### 📊 Medium Priority
- [ ] Multi-head ablation study
- [ ] Scaling experiments (100M+ params)
- [ ] Downstream task evaluation
- [ ] API documentation

### 📚 Low Priority
- [ ] Tutorials and examples
- [ ] Visualization tools
- [ ] Integration with Hugging Face
- [ ] Docker containers

## Questions?

- Open a [Discussion](https://github.com/yourusername/kolosis/discussions)
- Join our community chat (link TBD)
- Email: your.email@example.com

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to advance AI research together.

---

Thank you for contributing to Kolosis! 🚀
