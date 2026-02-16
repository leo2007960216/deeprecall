"""Tests for the deprecation utilities."""

import warnings

from deeprecall.core.deprecations import deprecated, warn_deprecated_param


class TestDeprecatedDecorator:
    def test_emits_deprecation_warning(self):
        @deprecated("Use new_func instead", "0.5.0")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func" in str(w[0].message)
            assert "0.5.0" in str(w[0].message)

    def test_includes_alternative(self):
        @deprecated("old api", "0.5.0", alternative="new_api")
        def old_api():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_api()
            assert "new_api" in str(w[0].message)

    def test_preserves_function_name(self):
        @deprecated("old", "0.5.0")
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_has_deprecated_attribute(self):
        @deprecated("old", "0.5.0")
        def my_function():
            pass

        assert hasattr(my_function, "__deprecated__")
        assert "deprecated" in my_function.__deprecated__.lower()


class TestWarnDeprecatedParam:
    def test_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_param("old_param")
            assert len(w) == 1
            assert "old_param" in str(w[0].message)

    def test_includes_alternative(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_param("old_param", alternative="new_param")
            assert "new_param" in str(w[0].message)

    def test_includes_removal_version(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_param("x", removal_version="1.0.0")
            assert "1.0.0" in str(w[0].message)
