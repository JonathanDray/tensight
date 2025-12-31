# Documentation Deployment Guide

## Local Development

Serve the documentation locally:

```bash
source venv/bin/activate
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

## Build Documentation

```bash
mkdocs build
```

The site will be generated in the `site/` directory.

## Deploy to GitHub Pages

### Automatic Deployment (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/docs.yml`) that automatically deploys the documentation when you push to the `main` branch.

To enable:
1. Go to your repository settings on GitHub
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "GitHub Actions"
4. The workflow will automatically deploy on the next push

### Manual Deployment

```bash
# Build the site
mkdocs build

# Deploy to gh-pages branch
mkdocs gh-deploy
```

## Custom Domain (Optional)

To use a custom domain:
1. Create a `CNAME` file in `docs/` with your domain name
2. Configure DNS settings as per GitHub Pages documentation

## Updating Documentation

1. Edit files in `docs/`
2. Commit and push to `main`
3. GitHub Actions will automatically rebuild and deploy

