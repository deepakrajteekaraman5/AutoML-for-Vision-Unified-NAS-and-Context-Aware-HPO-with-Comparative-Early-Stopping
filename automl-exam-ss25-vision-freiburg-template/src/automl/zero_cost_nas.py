# new file: src/automl/zero_cost_nas.py
import torch
from naswot.zc import synflow, jacob_cov, grad_norm


def compute_synflow_score(model, input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy_input = torch.ones(input_shape)
    return synflow(model, dummy_input)


def score_model(model_fn, method="synflow", input_shape=(1, 3, 224, 224)):
    model = model_fn()
    model.eval()
    if method == "synflow":
        return compute_synflow_score(model, input_shape)
    elif method == "jacobian":
        return jacob_cov(model, torch.ones(input_shape))
    elif method == "gradnorm":
        return grad_norm(model, torch.ones(input_shape))
    else:
        raise ValueError(f"Unsupported zero-cost method: {method}")
```
```
# modifications in src/automl/automl.py
@@ def _phase2_architecture_selection(self, architectures: Optional[List[str]]) -> List[str]:
-        # Use provided architectures or select strategically
+        # Use provided architectures or select strategically
@@
-        selected_architectures = architectures
+        selected_architectures = architectures
@@
-        self.logger.info(f"Selected architectures: {selected_architectures}")
+        self.logger.info(f"Selected architectures (pre-ranking): {selected_architectures}")
+        # ===== Zero-Cost NAS filter =====
+        try:
+            from .zero_cost_nas import score_model
+            zero_cost_method = self.config.get('zero_cost_method', 'synflow')
+            top_k = self.config.get('zero_cost_top_k', len(selected_architectures))
+            scores = []
+            for arch in selected_architectures:
+                model_fn = lambda name=arch: self.model_factory.create_model(name)
+                input_shape = (1,
+                               self.dataset_info['characteristics']['channels'],
+                               self.dataset_info['characteristics']['image_height'],
+                               self.dataset_info['characteristics']['image_width'])
+                score = score_model(model_fn, method=zero_cost_method, input_shape=input_shape)
+                scores.append((arch, score))
+                self.logger.debug(f"Zero-cost score for {arch}: {score:.4f}")
+            # select top-k by score
+            scores.sort(key=lambda x: x[1], reverse=True)
+            selected_architectures = [arch for arch, _ in scores[:top_k]]
+            self.logger.info(f"Architectures after Zero-Cost NAS filter (top {top_k}): {selected_architectures}")
+        except Exception as e:
+            self.logger.warning(f"Zero-Cost NAS filtering failed: {e}")
+        # ===== end zero-cost filter =====
```
