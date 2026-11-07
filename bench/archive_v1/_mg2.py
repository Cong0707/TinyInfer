import sys, torch, time, inspect
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.moe_gemv as MG
print('RESULT gate_up sig:', str(inspect.signature(MG.gate_up))[:140])
print('RESULT public:', [n for n,_ in inspect.getmembers(MG, inspect.isfunction) if not n.startswith('_')])