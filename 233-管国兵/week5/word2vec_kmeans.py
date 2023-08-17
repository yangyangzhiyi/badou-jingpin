#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


vector_size  = 100;
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    # 分词后的每行字符串，放入set()
    sentences = set() # {'彩民 周刊 专家组 双色球 09029 期 ： 防一区 弱   三区 热', '专家 解答 ： 通过 “ 双 录取 ” 可 读 何种 名校', '十大 公司 年终奖 揭秘   中海油 局级 领导 最高 30 万', '海军 建 世界 最 年轻 舰艇 编队   南海 周边 无敌', '台湾 首 证实 正研 无人 攻击机   欲 侦察 大陆 军用机场', '聚焦 汽车 智能化 争夺战   现有 应用 均 欠 完善', '奥博 托 重返 马刺 万人 起立 欢呼   姚明 走时 休城会 怎样', '年底 冲刺   10 种 减肥法 谁 最 牛 （ 组图 ）', '天津 一汽 借新 品牌 涅槃   全面 征战 小车 市场', '经纬 纺机 ： 控股 中融   业绩 爆发式 增长', '面包车 4 年 中 闯 118 次 红灯 被 查扣', '主卧 在 设计 时 很 注重 情调 （ 图 ）', '男篮 前 主帅 率队 击败 现任   这是 郭士强 最 想要 的 胜利', '古 石刻 现 “ 爸爸 去 哪儿 ”   南朝 古墓 3 辟邪 相 依偎', '话剧 《 公民 》 今日 首演   毛 俊杰 被 赞 实力派 演员', '日 企首 对 强征 美国 劳工 道歉   未涉 中国 劳工', '21 世纪 经济 报道 ： 2009   楼市 的 演进 与 回归', '来 澳洲 读 高中 的 真相 ： 别 被 中介 所惑 （ 图 ）', '末日 启示录 : 盘点 地球 上 那些 最 恐怖 的 坑洞', '宝哥 五 要素 解读 ： 盘口 预示 曼城 有 冷   不来梅 作稳胆', '国产汽车 “ 四大家族 ” 出击   再战 自主 品牌', '解密 电影 007 中 的 沙漠 避风港 （ 组图 ）', '中 韩 WCG 冠军 对决 ： 11 点护甲狮 鹫 龙鹰', '新浪 数字 彩票 频道 上线   开奖 图表 专家 媒体 尽收眼底', '叶 贞德 ： 打造 与 90 后 更 亲近 的 香港 玩法', '莫言 为 孔庙 题字 被 指 “ 题书错 向 ” 专家 : 应 遵 旧制', '主卧 空间 的 本质 讲究 精致 （ 图 ）', '艾怡良 新歌 歌词 被误 听   美丽 的 错误 促 良缘', '十一 将 至 多家 景区 门票 涨价   5A 景区 均价 破 百元', '1 元 钱 未 还 多付 800 元 利息   过年 刷卡 需 掌握 窍门', '哈佛 欲 在 网上 公开 10 位 知名 科学家 DNA 信息', '组图 ： 8 招   寒冬 穿成 轻便 瘦美人', '环 首都 经济圈 加热 燕郊 房市   品质 房乐居 优惠 5 万', '艾薇儿 的 帽衫 就 另类 的 多', '美国 最新 战舰 测试 航行 时速 突破 80 公里 （ 图 ）', '课堂 ： 这些 毛衣 搭 外套 最美 （ 组图 ）', '解放军 “ 超级 武器 ” 成功 在 即   可 猛增 军队 战力', '经适 房 限价 房 按 建筑面积 计价', '视频 亲民党 不满 陈水扁 柔性 政变 说 威胁 将 其 罢免', '向佐肩 扛 薛之谦遭 扒 衣   单手 射门 现 少林足球 奇迹', '《 露水 红颜 》 发 兽性 海报   刘亦菲 和 Rain 相爱 相杀', '专家 推荐 回顾 ： 久洋 256 中 8 场   刘建宏 凯利 防住 大冷', 'WCG 战报 ： 美兽 男 Lyn 九 分钟 终结 蛋 总', '上汽 大通 首款 MPV 下线     预计 北京 车展 上市', '英国 出现 百万只 星 椋鸟 齐飞 奇观 （ 组图 ）', '假借 分期付款 买车   信用卡 额度 1 万可 套现 20 万', '多家 银行 封杀 信用卡 支付宝 交易', '抓拍 ： 北京 胡同 MM 早春 装扮', '涉枪 疑犯 逃跑 闯进 部队 营区 被 哨兵 擒获', '流动性 资金 进入 商业地产 需要 正确引导 （ 图 ）', '葛剑雄 ： 恰如其分 地 拿捏 书生 意气 与 职业道德', '大兴 黄村 枣园 站 纯新 盘中 建 国际 港 即将 推出', '预防针 ： 留学生 须 警惕 英国 “ 野鸡大学 ”', '哪些 情况 不能 泡脚   注意事项 要 牢记', '鼻炎 治不好 怎么办   那 是 你 没试 过 这些 方法', '山东 十大 文化 旅游 目的地 品牌 推介会 在 京 启幕', '简直 可以 安放 灵魂   欧洲 这 几个 图书馆 太 美 （ 组图 ）', '留法 校长 的 人文 情怀 ： 川外 女生 不是 花瓶 （ 组图 ）', '保洁员 在 小区 垃圾箱 内 发现 被弃 男婴', '运势 最好 的 几种 骨相', '大商股份 ： 向 集团 收购 一处 物业', '侯明昊 宋 小宝 共舞   超 能 少年 开启 实力 霸屏 模式', '苏妙玲 《 蓝友 》 首播   出 道 三年 定制 粉丝 专属 主打', '湖南 沅陵 发现 古代 “ 数钞机 ”', '蓝色 港湾   老爸 向前 冲 VS 我家 大厨 挑战赛 （ 图 ）', '美韩 近期 将 举行 两次 大规模 军演   假想 解放军 入朝', '语文版 中小学 教材 “ 换血 ”   孩子 究竟 需要 怎样 的 教材 ？', '研究 称 6 万年前 走出 非洲 人类 祖先 男 多于 女', '房产 投资 专家 点评 北京 哪里 买房 最 赚钱 （ 图 ）', '花季 女子 称 兜售 初夜 救父   记者 暗访 发现 蹊跷', '专访 Dream 领队 小狼 ： 不 虚 任何 对手', 'PCGA2009 百事 超级 大奖赛 衡阳 赛区 开赛', '男童 打闹 时 被 衣棍 铁头 戳 进 前额', '胡敬雯 : 优化 旅游 体验 , 欢迎 更 多 中国 游客 到访 澳大利亚', '搭配 QA ： 夏季 应该 穿 什么 才 显瘦 呢 ？', '美 科学家 用 5 岁 女孩 卵巢 组织 培育出 卵子', '金銮殿 最早 出现 在 唐代   曾 只是 皇宫 中 一个 偏殿', '高颜值   +     实用 派         奔腾 B50 和 你 一起 出彩', '视频 ： 百万 富姐 高价 找 男人 生子   想 见面 先交 98 块', '台湾地区 户籍 人口 达 2300 万人', '2009 高端 物业 大奖 30 强 浮出 水面', '大族 激光 ： 谷底 已过   前途 光明', '韩国 航空 试验 中心 揭秘 ： 战斗机 被 冰冻 住 测试', '物业费 上涨 是否 需要 相关 部门 批准', '北京 40 论剑 阿拉善 英雄 会', '东南亚 风格 样板间   演绎 现代 时尚 （ 组图 ）', 'DIY   自己 动手 制作 精美壁纸 （ 组图 ）', '80 后 低资 高调 的 家装 家具 （ 图 ）', '" 后 旅人 时代 " 的 谷岳 ， 把 世界 迎进 Airbnb 的 家', '德系 双雄 C 级车 份额 争夺 升级 u3000 垄断 暂难 破', '经适 房 房主 上演 自买 自卖', '美韩 海军陆战队 在 白翎岛 举行 联合 军演 （ 图 ）', '一边 FUSION 一边 妖娆 （ 图 ）', '关 小刀 火线 推荐 ： 锡耶纳 抢分 保级   毕尔巴鄂 分 胜负', '英国 11 月 零售额 下降 0.4%', '抢个 篮板 差点 膝盖 报销   以为 要 退役 麦蒂 当场 泪 纵横', '搭配 QA ： 秋季 外套 咋 搭配 有 层次感 ？', '山东 ： 国民党 抗战 老兵 按 解放军 标准 给予 救助', '科学家 揭开 5000 年前 冰人 奥茨 身份 之谜 （ 图 ）', '盘口 任选 九场 ： 寻冷方能 成就 大奖   马赛 双选 更 稳妥'...
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences: # '南海舰队 司令 登 岛礁 检查 部队 战备   菲称 苗头 不 对'
        words = sentence.split()  #sentence是分好词的，空格分开 ： ['四大', '经典', '胎教', '法', '什么', '时候', '开始', '胎教']
        vector = np.zeros(model.vector_size) # 100维全0向量
        vector_size = model.vector_size
        #所有词的向量相加求平均，作为句子向量
        for word in words: # 南海舰队
            try:
                vector += model.wv[word] # 词，返回词向量
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words)) # 一句话的累加词向量 除以 词个数，放入返回的np.array里
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    # [[-0.15490893  0.2279559  -0.02212039 ...  0.23468943 -0.09360183,   0.2159754 ], [-0.14795631  0.08143186  0.40312179 ...  0.05402408  0.12285135,   0.17269374], [-0.2927731   0.02531278  0.21290847 ...  0.058404    0.11072411,   0.22417384], ..., [-0.430
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量，句子总数取平方根，结果为聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 所有词向量按lable分类: {'0': {[],[]...}, '1': {[],[]...}}
    lable_vc_dict = {}
    for i in range(n_clusters):
        lable_i_vc = []
        for l, v in zip(kmeans.labels_, vectors):
            if i == l:
                lable_i_vc.append(v)
        lable_vc_dict[i] = lable_i_vc

    # 每个lable的平均类内距离
    lable_center_avg_distant = {} # {0: 0.2123, 1: 0.1233 ...}
    index = 0
    for center in kmeans.cluster_centers_:
        avg_distant = 0.0
        for vc in lable_vc_dict[index]:
            avg_distant += np.sqrt(np.sum((vc - center)**2)) # 离质心距离，累加
        lable_center_avg_distant[index] = avg_distant / len(lable_vc_dict[index]) # 累加的质心距离求平均
        index += 1
    # 排序：最短平均类内距离在前
    lable_center_avg_distant_sorted = sorted(lable_center_avg_distant.items(),
                                      key=lambda x: x[1], reverse=False)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    class_num = 10 # 打印10份聚类结果
    print("---类内距离最近%d类------" % class_num)
    for i in range(min(class_num, len(lable_center_avg_distant_sorted))):
        print("\n---------%d/%d lable: %d 类内平均距离：%f" % (i+1, class_num, lable_center_avg_distant_sorted[i][0], lable_center_avg_distant_sorted[i][1]))
        for t in zip(sentence_label_dict[lable_center_avg_distant_sorted[i][0]]):
            print(t[0].replace(" ", ""))

if __name__ == "__main__":
    main()

