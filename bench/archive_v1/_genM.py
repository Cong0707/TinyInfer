import sys, os
sys.argv = ["generate_kernels.py", "True", r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\quantization\marlin"]
exec(open(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\quantization\marlin\generate_kernels.py").read().replace('if __name__ == "__main__":', 'if False:'))
SUPPORT_SM80 = True
generate_new_kernels()
print("generated")
