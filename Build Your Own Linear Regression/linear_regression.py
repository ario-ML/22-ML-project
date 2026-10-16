"""
Build Your Own Linear Regression
==================================

پیاده‌سازی رگرسیون خطی از صفر، با دو روش بهینه‌سازی:
    1. Gradient Descent  (batch)
    2. Normal Equation   (حل تحلیلی بسته)

هر دو روش از رگولاریزیشن L2 (Ridge) پشتیبانی می‌کنند.
API شبیه sklearn طراحی شده: fit / predict / score
"""

import numpy as np


# ----------------------------------------------------------------------
# توابع کمکی
# ----------------------------------------------------------------------

def add_intercept(X: np.ndarray) -> np.ndarray:
    """یک ستون از جنس ۱ به ابتدای X اضافه می‌کند (برای بایاس/عرض از مبدا)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """میانگین مربعات خطا."""
    return float(np.mean((y_true - y_pred) ** 2))


class StandardScaler:
    """
    استانداردسازی فیچرها: میانگین صفر، انحراف معیار یک.
    برای Gradient Descent حیاتیه، چون بدون این کار مقیاس فیچرها
    باعث می‌شه سطح خطا (loss surface) خیلی کشیده بشه و GD یا واگرا
    بشه یا خیلی کند همگرا بشه.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # جلوگیری از تقسیم بر صفر برای فیچرهای ثابت
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------
# کلاس اصلی
# ----------------------------------------------------------------------

class LinearRegression:
    """
    رگرسیون خطی با قابلیت انتخاب روش بهینه‌سازی.

    پارامترها
    ----------
    method : {'gradient_descent', 'normal_equation'}
        روش آموزش مدل.
    alpha : float
        ضریب رگولاریزیشن L2 (Ridge). صفر یعنی بدون رگولاریزیشن.
    learning_rate : float
        نرخ یادگیری (فقط برای gradient_descent).
    n_iterations : int
        تعداد تکرارها (فقط برای gradient_descent).
    standardize : bool
        اگر True باشه، فیچرها قبل از آموزش استاندارد می‌شن و
        ضرایب نهایی به مقیاس اصلی برگردونده می‌شن.
    tol : float
        اگر تغییر loss بین دو iteration کمتر از این مقدار بشه،
        gradient descent زودتر متوقف می‌شه (early stopping).
    """

    def __init__(
        self,
        method: str = "gradient_descent",
        alpha: float = 0.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        standardize: bool = True,
        tol: float = 1e-8,
        verbose: bool = False,
    ):
        if method not in ("gradient_descent", "normal_equation"):
            raise ValueError("method باید 'gradient_descent' یا 'normal_equation' باشه")

        self.method = method
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.standardize = standardize
        self.tol = tol
        self.verbose = verbose

        # بعد از fit پر می‌شن
        self.coef_ = None          # ضرایب فیچرها (بدون بایاس)
        self.intercept_ = None     # عرض از مبدا
        self.loss_history_ = []    # فقط برای gradient_descent
        self._scaler = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.standardize:
            self._scaler = StandardScaler()
            X_proc = self._scaler.fit_transform(X)
        else:
            X_proc = X

        if self.method == "normal_equation":
            self._fit_normal_equation(X_proc, y)
        else:
            self._fit_gradient_descent(X_proc, y)

        return self

    # ------------------------------------------------------------------
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        حل بسته‌ی معادله‌ی نرمال (Ridge):
            theta = (X^T X + alpha * I)^-1 X^T y
        نکته: ستون بایاس رگولاریزه نمی‌شه (سطر/ستون اول ماتریس I صفر می‌شه).
        """
        Xb = add_intercept(X)
        n_features_plus_bias = Xb.shape[1]

        reg_matrix = self.alpha * np.eye(n_features_plus_bias)
        reg_matrix[0, 0] = 0.0  # بایاس رگولاریزه نمی‌شه

        # از pinv به‌جای inv استفاده می‌کنیم تا اگه X^TX منفرد (singular) بود
        # هم کد خطا نده و پایدارتر باشه.
        theta = np.linalg.pinv(Xb.T @ Xb + reg_matrix) @ Xb.T @ y

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    # ------------------------------------------------------------------
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        theta = np.zeros(n_features)
        bias = 0.0

        self.loss_history_ = []
        prev_loss = np.inf

        for i in range(self.n_iterations):
            y_pred = X @ theta + bias
            error = y_pred - y

            # گرادیان‌ها (میانگین روی نمونه‌ها) + ترم رگولاریزیشن روی وزن‌ها
            grad_w = (X.T @ error) / n_samples + (self.alpha / n_samples) * theta
            grad_b = np.mean(error)

            theta -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

            loss = mse(y, y_pred) + (self.alpha / n_samples) * np.sum(theta ** 2)
            self.loss_history_.append(loss)

            if self.verbose and i % max(1, self.n_iterations // 10) == 0:
                print(f"iter {i:5d}  loss={loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"همگرایی زودهنگام در iteration {i} (loss تغییر نکرد)")
                break
            prev_loss = loss

        self.coef_ = theta
        self.intercept_ = bias

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("مدل هنوز fit نشده. اول fit(X, y) رو صدا بزن.")

        X = np.asarray(X, dtype=float)
        X_proc = self._scaler.transform(X) if self.standardize else X
        return X_proc @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ضریب تعیین R² : هرچی به ۱ نزدیک‌تر بهتر."""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    # ------------------------------------------------------------------
    def get_params_readable(self):
        """چاپ خوانای ضرایب مدل (بعد از fit)."""
        return {"intercept": self.intercept_, "coef": self.coef_}
