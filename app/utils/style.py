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

    html, body, h1, code, [class*="st-"] {
        font-family: 'UnistraA', sans-serif !important;
    }

    .block-container {
        padding-top: 3.75rem;
    }

    html {
        font-size: 18px !important;
    }

    label, label > div, code, pre, small {
        font-size: 16px !important;
    }
    </style>
    """
