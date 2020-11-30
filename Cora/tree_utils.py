def getEntitiesID(filepath):
    nums = []
    with open(filepath,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            nums.append(line[0])
    idx_dict = {}
    for idx,id in enumerate(nums):
        idx_dict[id] = idx
    return idx_dict
def getweightDicts(filepath,idx_dict):
    adj_dicts = {}
    with open(filepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        cnt = adj_dicts.get(line[0],-1)
        if cnt == -1:
            adj_dicts[line[0]] = [line[1]]
        else:
            if line[1] not in adj_dicts[line[0]]:
                adj_dicts[line[0]].append(line[1])
        cnt = adj_dicts.get(line[1],-1)
        if cnt == -1:
            adj_dicts[line[1]] = [line[0]]
        else:
            if line[0] not in adj_dicts[line[1]]:
                adj_dicts[line[1]].append(line[0])
    weight_dict = {}
    for key,value in adj_dicts.items():
        for item in value:
            cnt = weight_dict.get(key,-1)
            if cnt == -1:
                weight_dict[key] = {}
            weight_dict[key][item] = 1.0
    for key,value in weight_dict.items():
        for item in value:
            for key1,value1 in weight_dict.items():
                if key in value1 and item in value1 and key != key1 and item != key1:
                    weight_dict[key][item] += 1.0
                    weight_dict[item][key] += 1.0
        weight_dict[key][key] = 100.0
    final_dict = {}
    for key,value in weight_dict.items():
        final_dict[idx_dict[key]] = {}
        for item,num in value.items():
            final_dict[idx_dict[key]][idx_dict[item]] = num
    return final_dict

def sortNodesall(weight_dicts):
    huf_idx = {}
    for key,weight_dict in weight_dicts.items():
        node_idx = []
        weight_dict = sorted(weight_dict.items(),key=lambda  x:x[1],reverse=True)
        if(len(weight_dict) == 1):
            weight_dict.append(weight_dict[0])
        while len(weight_dict) != 1:
            node_1 = weight_dict.pop()
            node_2 = weight_dict.pop()
            node_idx.append(node_1[0])
            node_idx.append(node_2[0])
            out = ('unknown',node_1[1]+node_2[1])
            weight_dict.append(out)
            weight_dict = sorted(weight_dict,key=lambda x:x[1],reverse=True)
        while True:
            if "unknown" in node_idx:
                node_idx.remove("unknown")
            else:
                break
        huf_idx[key] = node_idx
    return huf_idx

def sortNodes(weight_dict):
    weight_dict = sorted(weight_dict.items(),key=lambda x:x[1],reverse=False)
    node_list = [item[0] for item in weight_dict]
    return node_list

def createDict(file1,file2):
    idx_dict = getEntitiesID(file1)
    weight_dict = getweightDicts(file2,idx_dict)
    return weight_dict