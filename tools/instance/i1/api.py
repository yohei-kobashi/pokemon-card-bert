import importlib, inspect, pkgutil
import onnxruntime.quantization as Q
print("modules:", sorted(m.name for m in pkgutil.iter_modules(Q.__path__)))
for mod in ("onnxruntime.quantization.matmul_nbits_quantizer",
            "onnxruntime.quantization.matmul_bnb4_quantizer"):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(mod, "IMPORT FAIL", type(e).__name__, e); continue
    print("\n###", mod)
    print("names:", [n for n in dir(m) if not n.startswith("_")])
    for n in dir(m):
        o = getattr(m, n)
        if inspect.isclass(o) and ("Config" in n or "Quantizer" in n):
            try: print(" ", n, inspect.signature(o.__init__))
            except Exception: print(" ", n, "(no sig)")
