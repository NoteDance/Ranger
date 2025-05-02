# Ranger

**Overview**:

The `Ranger` optimizer combines the benefits of RAdam (Rectified Adam) and LookAhead optimizers to improve training stability, convergence, and generalization. It incorporates gradient centralization (GC) for both convolutional and fully connected layers and uses an integrated LookAhead mechanism to smooth updates.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay. Applies standard weight decay during updates.
- **`alpha`** *(float, default=0.5)*: LookAhead interpolation factor, determining how much to interpolate between fast and slow weights.
- **`k`** *(int, default=6)*: Number of optimizer steps before LookAhead updates.
- **`N_sma_threshhold`** *(int, default=5)*: Threshold for the simple moving average (SMA) used in RAdam to enable variance rectification.
- **`use_gc`** *(bool, default=True)*: Whether to apply gradient centralization (GC).
- **`gc_conv_only`** *(bool, default=False)*: Whether to apply GC only to convolutional layers (`True`) or to all layers (`False`).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="ranger")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from ranger import Ranger

# Instantiate optimizer
optimizer = Ranger(
    learning_rate=1e-3,
    weight_decay=1e-2,
    alpha=0.5,
    k=6,
    use_gc=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger2020

**Overview**:

The `Ranger` optimizer combines the techniques of RAdam (Rectified Adam) and LookAhead to achieve faster convergence and better generalization. It also optionally incorporates Gradient Centralization (GC), which re-centers the gradient to improve optimization stability.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay.
- **`alpha`** *(float, default=0.5)*: Interpolation factor for the LookAhead mechanism.
- **`k`** *(int, default=6)*: Number of update steps before LookAhead interpolates weights.
- **`N_sma_threshhold`** *(int, default=5)*: Threshold for the simple moving average (SMA) in RAdam to apply rectified updates.
- **`use_gc`** *(bool, default=True)*: Whether to apply Gradient Centralization (GC).
- **`gc_conv_only`** *(bool, default=False)*: If `True`, GC is only applied to convolutional layers.
- **`gc_loc`** *(bool, default=True)*: If `True`, GC is applied during the gradient computation step; otherwise, it is applied after.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="ranger2020")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from ranger2020 import Ranger

# Instantiate optimizer
optimizer = Ranger(
    learning_rate=1e-3,
    alpha=0.5,
    k=6,
    use_gc=True,
    gc_conv_only=False
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# RangerVA

**Overview**:

The `RangerVA` optimizer is a hybrid optimizer that combines the techniques of Rectified Adam (RAdam), Lookahead optimization, and gradient transformations, making it suitable for modern deep learning tasks. It also includes support for custom gradient transformations and adaptive learning rate calibration, making it highly flexible and efficient for various scenarios.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay.
- **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor controlling the update between fast and slow weights.
- **`k`** *(int, default=6)*: Number of steps before Lookahead updates are applied.
- **`n_sma_threshhold`** *(int, default=5)*: Threshold for the number of simple moving averages in RAdam's variance rectification mechanism.
- **`amsgrad`** *(bool, default=True)*: Whether to use the AMSGrad variant.
- **`transformer`** *(str, default='softplus')*: Specifies the transformation function applied to the adaptive learning rate (e.g., `'softplus'` for smooth adaptation).
- **`smooth`** *(float, default=50)*: Smoothing factor for the Softplus transformation function.
- **`grad_transformer`** *(str, default='square')*: Specifies the transformation applied to gradients (e.g., `'square'` or `'abs'`).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="rangerva")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from rangerva import RangerVA

# Instantiate optimizer
optimizer = RangerVA(
    learning_rate=1e-3,
    alpha=0.5,
    k=6,
    weight_decay=1e-2,
    transformer='softplus',
    smooth=50,
    grad_transformer='square'
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# RangerQH

**Overview**:

The `RangerQH` optimizer combines the QHAdam optimization algorithm with the Lookahead mechanism. QHAdam, introduced by Ma and Yarats (2019), balances the contributions of gradients and gradient variances with tunable parameters `nu1` and `nu2`. The Lookahead mechanism, developed by Hinton and Zhang, enhances convergence by interpolating between "fast" and "slow" weights over multiple steps. This combination provides a powerful optimization approach for deep learning tasks, offering improved convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Supports both standard and decoupled decay.
- **`nus`** *(tuple, default=(0.7, 1.0))*: The `nu1` and `nu2` parameters controlling the blending of gradient and variance components.
- **`k`** *(int, default=6)*: Number of optimization steps before updating "slow" weights in the Lookahead mechanism.
- **`alpha`** *(float, default=0.5)*: Interpolation factor for Lookahead updates.
- **`decouple_weight_decay`** *(bool, default=False)*: Enables decoupled weight decay as described in AdamW.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="rangerqh")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from ranger_qh import RangerQH

# Instantiate optimizer
optimizer = RangerQH(
    learning_rate=1e-3,
    nus=(0.8, 1.0),
    k=5,
    alpha=0.6,
    weight_decay=1e-2,
    decouple_weight_decay=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
