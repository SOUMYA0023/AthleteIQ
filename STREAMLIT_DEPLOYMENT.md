# Streamlit Cloud Deployment Guide

## 🚀 Deployment Steps

### 1. Repository Setup
- ✅ Code is in GitHub: `https://github.com/SOUMYA0023/AthleteIQ.git`
- ✅ `requirements.txt` is at the root of the repository
- ✅ App path: `sports_form_analysis/app/app.py`

### 2. Streamlit Cloud Configuration

When deploying on Streamlit Cloud:

1. **App Path**: Set to `sports_form_analysis/app/app.py`
2. **Python Version**: 3.10 or higher
3. **Dependencies**: Automatically installed from `requirements.txt`

### 3. Required Files (Already Created)

- ✅ `requirements.txt` - Python dependencies (at root)
- ✅ `packages.txt` - System packages for MediaPipe/OpenCV (at root)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `sports_form_analysis/models/__init__.py` - Package init
- ✅ `sports_form_analysis/utils/__init__.py` - Package init

### 4. Model Training

**Important**: The model file (`form_classifier.pkl`) should be committed to the repository for Streamlit Cloud.

If the model is not in the repo, you have two options:

**Option A: Commit the model (Recommended)**
```bash
# The model is already in .gitignore exception
git add sports_form_analysis/models/form_classifier.pkl
git commit -m "Add trained model for deployment"
git push
```

**Option B: Train on first run**
Add this to the app to train if model doesn't exist:
```python
if not os.path.exists(model_path):
    with st.spinner("Training model (first time only)..."):
        import subprocess
        subprocess.run(['python', 'sports_form_analysis/models/train_model.py'])
```

### 5. Deployment Checklist

- [x] `requirements.txt` at repository root
- [x] `packages.txt` at repository root (for system dependencies)
- [x] App path correctly set: `sports_form_analysis/app/app.py`
- [x] All imports use relative paths
- [x] Model file accessible (committed or trainable)
- [x] `__init__.py` files in packages

### 6. Common Issues & Solutions

#### Issue: ModuleNotFoundError for cv2
**Solution**: 
- Ensure `opencv-python` is in `requirements.txt`
- Check `packages.txt` has system dependencies

#### Issue: ModuleNotFoundError for models
**Solution**: 
- Verify `__init__.py` files exist in `models/` and `utils/`
- Check import paths in `app.py`

#### Issue: Model not found
**Solution**: 
- Commit `form_classifier.pkl` to repository
- Or implement on-demand training (Option B above)

#### Issue: Path errors
**Solution**: 
- The app now handles both local and Streamlit Cloud paths
- Uses `Path` objects for cross-platform compatibility

### 7. Testing Deployment

After deploying:

1. Check Streamlit Cloud logs for errors
2. Verify all dependencies install correctly
3. Test video upload functionality
4. Verify model loads successfully

### 8. Environment Variables (if needed)

If you need to set environment variables in Streamlit Cloud:
- Go to app settings
- Add environment variables in "Secrets" section

---

## 📝 Files Modified for Deployment

1. **`requirements.txt`** - Moved to root, all dependencies listed
2. **`packages.txt`** - Created for system packages
3. **`app/app.py`** - Updated import paths for Streamlit Cloud
4. **`.streamlit/config.toml`** - Streamlit configuration
5. **`models/__init__.py`** - Package initialization
6. **`utils/__init__.py`** - Package initialization
7. **`.gitignore`** - Updated to allow model file

---

## ✅ Deployment Ready!

Your app should now deploy successfully on Streamlit Cloud. The main changes ensure:
- Dependencies are correctly specified
- Import paths work in cloud environment
- Model file is accessible
- All packages are properly initialized

