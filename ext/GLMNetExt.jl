module GLMNetExt


"""
This file is just a small placeholder so we have a tidy spot to add
GLMNet-specific glue later. 

What to do when I'm ready:
- add `GLMNet` to the environment
- Do `using GLMNet; using TMLE` and TMLE will pick up the extra bits
	(Lasso strategy and MLJ wrappers) via conditional loading.

Leaving this here so the extension structure matches the other `ext/` files.
I can update or remove it whenever you want — keeping it handy for later.
"""

using GLMNet
using TMLE

end
