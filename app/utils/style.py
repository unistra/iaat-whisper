def custom_font() -> str:
    """
    Load the UnistraA font from the static folder
    """

    return """
    <style>
    @font-face {
        font-family: 'UnistraA';
        src: url('/app/static/fonts/UnistraA-Regular.ttf') format('truetype');
    }

    @font-face {
        font-family: 'UnistraA';
        src: url('/app/static/fonts/UnistraA-Bold.ttf') format('truetype');
        font-weight: bold;
    }

    @font-face {
        font-family: 'UnistraA';
        src: url('/app/static/fonts/UnistraA-Italic.ttf') format('truetype');
        font-style: italic;
    }

    @font-face {
        font-family: 'UnistraA';
        src: url('/app/static/fonts/UnistraA-BoldItalic.ttf') format('truetype');
        font-weight: bold;
        font-style: italic;
    }

    /* Appliquer la police partout sauf sur les icônes */
    html, body, h1, code, [class*="st-"] {
        font-family: 'UnistraA', sans-serif !important;
    }

    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    .block-container {
        padding-top: 3.75rem;
    }

    html, span[label] {
        font-size: 18px !important;
    }

    label, label > div, code, pre, small {
        font-size: 16px !important;
    }
    </style>
    """
