# -*- coding: utf-8 -*-

from config import num_joints, idx_in_raw, idx_in_raw_str, joint_pairs

if __name__ == '__main__':
    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for j in range(num_joints):
        key = '$(body_part_{})'.format(j)
        value = idx_in_raw_str[idx_in_raw.index(j)]
        template = template.replace(key, value)
        key = '$(joint_pair_{})'.format(j)
        j0 = joint_pairs[j][0]
        v0 = idx_in_raw_str[idx_in_raw.index(j0)]
        j1 = joint_pairs[j][1]
        v1 = idx_in_raw_str[idx_in_raw.index(j1)]
        value = '{}->{}'.format(v0, v1)
        template = template.replace(key, value)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
