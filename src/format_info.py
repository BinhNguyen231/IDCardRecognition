def format_character_special(info):
    table = str.maketrans(dict.fromkeys('!"#$%&\'()*+;<=>?@[\\]^_`{|}~'))
    info = info.translate(table)
    info = info.split(':')[-1]
    return info

def format_space(info):
    return " ".join(info.split())

def preprocessing(info):
    table = str.maketrans(dict.fromkeys('!"$%&\()*+,-./:;<=>?@[\\]^_`{|}~'))
    info = str(info).translate(table)
    info = info.lower()
    info = format_character_special(info)
    info = format_space(info)
    return info

def format_gioi_tinh(info):
    info = info.lower()
    if 'emp' in info:
        return 'empty'
    elif 'na' in info:
        return 'Nam'
    else:
        return 'Nữ'

def format_ID(info):
    return format_space(info)

def format_ho_ten(info):
    info = format_space(info)
    return info.title()

def format_ngay_sinh(info):
    return format_space(info)

def format_quoc_tich(info):
    info = info.lower()
    if 'vi' in info or 'na' in info:
        return 'Việt Nam'
    else:
        info = info.split(':')[-1]
        info = format_space(info)
        info = info.title()
        return info
        

def format_dan_toc(info):
    info = info.lower()
    if 'ki' in info:
        return 'Kinh'
    elif 'emp' in info:
        return 'empty'
    else:
        info = info.split(':')[-1]
        info = format_space(info)
        info = info.title()
        return info
    
def format_diachi_quequan(info):
    if info.startswith('Xã'):
        info = info[3:]
    return format_character_special(info)

def format_diachi_quequan_full(info0, info1):
    if info0 == '':
        info = info1
    elif info1 =='':
        info = info0
    else:
        info = info0 +', '+ info1
    info = format_character_special(info)
    if ',,' in info:
        info = info.replace(',,', ',')
    return info

def format_hsd(info):
    info = info.lower()
    if 'kh' in info:
        return 'Không thời hạn'
    else:
        info = info.split(' ')[-1]
        table = str.maketrans(dict.fromkeys('dđêếễểnéẽềè'))
        info = info.translate(table)
        info = info.split(':')[-1]
        return info


def format_information(d):
    classes = ['id', 'ho_ten', 'ngay_sinh', 'gioi_tinh', 'quoc_tich', 'dan_toc', 
                'que_quan_0', 'que_quan_1',
                'noi_thuong_tru_0', 'noi_thuong_tru_1', 'hsd']
    info_dict = dict()
    output_dict = dict()
    for c in classes:
        if c in d.keys():
            info_dict[c] = d[c]
        else:
            info_dict[c] = ''
    output_dict['id'] = format_ID(info_dict['id'])
    output_dict['ho_ten'] = format_ho_ten(info_dict['ho_ten'])
    output_dict['ngay_sinh'] = format_ngay_sinh(info_dict['ngay_sinh'])
    output_dict['gioi_tinh'] = format_gioi_tinh(info_dict['gioi_tinh'])
    if info_dict['quoc_tich'] != '':
        output_dict['quoc_tich'] = format_quoc_tich(info_dict['quoc_tich'])
    if info_dict['dan_toc'] != '':
        output_dict['dan_toc'] = format_dan_toc(info_dict['dan_toc'])
    output_dict['que_quan'] = format_diachi_quequan_full(info_dict['que_quan_0'], info_dict['que_quan_1'])
    output_dict['noi_thuong_tru'] = format_diachi_quequan_full(info_dict['noi_thuong_tru_0'], info_dict['noi_thuong_tru_1'])
    output_dict['hsd'] = format_hsd(info_dict['hsd'])
    return output_dict