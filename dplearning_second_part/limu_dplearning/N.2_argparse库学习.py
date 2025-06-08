import argparse # 用以接受命令行参数
# 解析器 必要
args=argparse.ArgumentParser(
    description="脚本描述",  # 显示在帮助信息中
    epilog="示例用法..."  # 帮助信息的结尾说明 # 这俩参数可选
)


# - 和--不一样
args.add_argument('-a',type=int,default=3,help='测试')
args.add_argument('--a',type=int,default=3,help='测试')
# 访问他们都是arg.a 所以上述这种写法有冲突
#建议下面这种写法
args.add_argument('-a','--alpha',type=int,default=3,help='测试')
arg=args.parse_args()
print(arg)