from streamlit.testing.v1 import AppTest
import pytest

def test_app():
    app = AppTest.from_file("../app/Accueil.py")
    app.run(timeout=10)
    assert not app.exception
    assert any("🔑 Se connecter avec votre compte universitaire" in btn.label for btn in app.button)


@pytest.mark.filterwarnings("ignore::Warning")
def test_transcription():
    app = AppTest.from_file("../app/pages/1_Transcription.py")
    app.run(timeout=10)
    assert not app.exception
    assert any("🔑 Se connecter avec votre compte universitaire" in btn.label for btn in app.button)


def test_sous_titrage():
    app = AppTest.from_file("../app/pages/2_Sous-titrage.py")
    app.run(timeout=10)
    assert not app.exception
    assert any("🔑 Se connecter avec votre compte universitaire" in btn.label for btn in app.button)
