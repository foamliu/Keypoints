# -*- coding: utf-8 -*-

from config import num_joints, idx_in_raw, idx_in_raw_str

if __name__ == '__main__':
    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for j in range(num_joints):
        key = '$(body_part_{})'.format(j)
        value = idx_in_raw_str[idx_in_raw.index(j)]
        template = template.replace(key, value)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
