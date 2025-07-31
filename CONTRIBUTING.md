# Contributing to ADTC Smart Crop Disease Classifier

Thank you for your interest in contributing to our AI-powered crop disease detection project! We welcome contributions from developers, agricultural experts, and data scientists.

## 🚀 Getting Started

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0)
- Git
- Basic knowledge of Kotlin/Android development
- For AI model work: Python 3.7+, TensorFlow 2.x

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/adtc-crop-disease-classifier.git
   cd adtc-crop-disease-classifier
   ```
3. Open in Android Studio
4. Build and run the project

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include device information, Android version, and steps to reproduce
- Attach screenshots or logs when helpful

### Feature Requests
- Open an issue with the "enhancement" label
- Describe the feature and its agricultural use case
- Explain how it would benefit farmers

### Code Contributions

#### 1. Choose an Area
- **🌾 New Crop Support**: Add training data for additional crops
- **🔬 Model Improvements**: Enhance accuracy and performance
- **📱 UI/UX**: Improve user interface and experience
- **🌍 Localization**: Add support for multiple languages
- **📊 Analytics**: Enhanced reporting and insights
- **🧪 Testing**: Add unit tests and integration tests

#### 2. Development Process
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `./gradlew test`
5. Update documentation if needed
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request

#### 3. Code Standards
- Follow Kotlin coding conventions
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused
- Follow Material Design guidelines for UI

#### 4. Commit Messages
Use clear, descriptive commit messages:
```
feat: add support for corn disease detection
fix: resolve camera permission crash on Android 11
docs: update training guide with new dataset
test: add unit tests for image preprocessing
```

### AI Model Contributions

#### Adding New Crops
1. Gather high-quality training images (minimum 1000 per class)
2. Follow the PlantVillage dataset structure
3. Update the training notebooks
4. Test the model accuracy
5. Update the supported crops documentation

#### Model Improvements
1. Experiment with different architectures
2. Optimize for mobile performance
3. Maintain or improve accuracy
4. Document your approach and results
5. Provide before/after performance comparisons

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code builds successfully
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Screenshots included for UI changes
- [ ] Performance impact assessed

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Tested on multiple devices

## Screenshots (if applicable)
Add screenshots for UI changes

## Agricultural Impact
Explain how this benefits farmers
```

## 🌾 Agricultural Domain Knowledge

### Understanding Crop Diseases
- Research common diseases for target crops
- Understand visual symptoms and progression
- Consider regional variations and climate factors
- Validate with agricultural experts when possible

### Data Quality Standards
- Images should be clear and well-lit
- Focus on diseased areas
- Include various disease stages
- Avoid heavily processed or filtered images
- Maintain consistent image quality across classes

## 🧪 Testing Guidelines

### Manual Testing
1. **Real Crop Testing**: Test with actual plants when possible
2. **Screen Testing**: Use high-quality crop images from research databases
3. **Edge Cases**: Test with non-crop objects, poor lighting, etc.
4. **Device Testing**: Test on various Android devices and versions

### Automated Testing
- Write unit tests for new functions
- Add integration tests for major features
- Include performance benchmarks
- Test model accuracy with validation datasets

## 📚 Documentation

### Code Documentation
- Add KDoc comments for public functions
- Document complex algorithms
- Include usage examples
- Update README for new features

### User Documentation
- Update user guides for new features
- Include screenshots and step-by-step instructions
- Consider different user skill levels
- Translate important documentation when possible

## 🌍 Localization

### Adding New Languages
1. Create new string resource files
2. Translate all user-facing text
3. Consider cultural context for agricultural terms
4. Test with native speakers when possible
5. Update language selection in settings

### Translation Guidelines
- Use agricultural terminology appropriate for the region
- Keep technical terms consistent
- Consider text length variations in UI layouts
- Provide context for translators

## 🏆 Recognition

### Contributors
All contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

### Significant Contributions
Major contributors may be invited to:
- Join the core development team
- Present at agricultural technology conferences
- Collaborate on research publications

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Technical questions and bug reports
- **Discussions**: General questions and feature discussions
- **Email**: Direct contact for sensitive issues

### Mentorship
New contributors can request mentorship for:
- Android development guidance
- Machine learning model development
- Agricultural domain knowledge
- Open source best practices

## 🔒 Security

### Reporting Security Issues
- Do not open public issues for security vulnerabilities
- Email security concerns directly to the maintainers
- Include detailed reproduction steps
- Allow time for fixes before public disclosure

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping us build better agricultural technology for farmers worldwide! 🌱**