import pickle
import re
import os

import ipdb

def main():
    """
    This function combines the exp+gain targets from the gridsearch method and feature method of label generation.
    """
    method1_path = '../data/gridsearch/'
    method2_path = '../data/features/'
    pfiles1 = next(os.walk(method1_path))[2]
    pfiles2 = next(os.walk(method2_path))[2]
    
    pfiles1.sort()
    pfiles2.sort()

    for i in range(len(pfiles1)):
        pfile1 = pickle.load(open(os.path.join(method1_path, pfiles1[i]), 'rb'))
        pfile2 = pickle.load(open(os.path.join(method2_path, pfiles2[i]), 'rb'))

        name = pfiles1[i]
        print("Generating new targets for \"{}\"".format(name))

        new_targets = []

        for j in range(len(pfile1)):
            target1 = pfile1[j]
            target2 = pfile2[j]

            params1 = [int(num) for num in re.findall(r'\d+', target1)]
            params2 = [int(num) for num in re.findall(r'\d+', target2)]
            gain1 = round(params1[3] + params1[4]/100, 2)
            gain2 = round(params2[3] + params2[4]/100, 2)
            exp1 = params1[2]
            exp2 = params2[2]

            imNum = params1[1]
            dataNum = params1[0]

            # Grid weighted
            # new_exp = int((exp1+exp1+exp2)/3)
            # new_gain = round((gain1+gain1+gain2)/3, 2)

            # Equal
            new_exp = int((exp1+exp2)/2)
            new_gain = round((gain1+gain2)/2, 2)

            new_target = os.path.join('data_{}_{}_exp-{}_gain-{}.jpg'.format(dataNum, str(imNum).zfill(4), new_exp, new_gain))

            new_targets.append(new_target)
        
        # save the new pfile
        os.makedirs('../data/hybrid_equal', exist_ok=True)
        filename = '../data/hybrid_equal/'+name
    
        f = open(filename, 'wb')
        pickle.dump(new_targets, f)
        f.close()

if __name__ == '__main__':
    main()