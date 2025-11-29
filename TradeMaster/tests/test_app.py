import unittest
from app import app  # assuming app is your Dash instance

class TestApp(unittest.TestCase):
    def test_app_has_layout(self):
        self.assertIsNotNone(app.layout, "Dash app should have a layout defined")

    def test_app_title(self):
        # If you set app.title somewhere
        self.assertTrue(hasattr(app, "title"), "App should have a title attribute")
