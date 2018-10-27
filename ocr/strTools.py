# -*- coding: utf-8  -*-

def is_chinese(usen):
        """判断一个unicode是否是汉字"""
        for uchar in usen:
            if uchar >= '\u4e00' and uchar<='\u9fa5':
                    continue
            else:
                    return False
        else:
            return True

def is_number(usen):
        """判断一个unicode是否是数字"""
        for uchar in usen:
            if uchar >= '\u0030' and uchar<='\u0039':
                    continue
            else:
                    return False
        else:
            return True

def is_alphabet(usen):
        """判断一个unicode是否是英文字母"""
        for uchar in usen:
                if (uchar >= '\u0041' and uchar<='\u005a') or (uchar >= '\u0061' and uchar<='\u007a'):
                       continue
                else:
                        return False
        else:
                return True
def is_other(uchar):
        """判断是否非汉字，数字和英文字符"""
        if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
                return True
        else:
                return False
def B2Q(uchar):
        """半角转全角"""
        inside_code=ord(uchar)
        if inside_code<0x0020 or inside_code>0x7e:      #不是半角字符就返回原来的字符
                return uchar
        if inside_code==0x0020: #除了空格其他的全角半角的公式为:半角=全角-0xfee0
                inside_code=0x3000
        else:
                inside_code+=0xfee0
        return chr(inside_code)
def Q2B(uchar):
        """全角转半角"""
        inside_code=ord(uchar)
        if inside_code==0x3000:
                inside_code=0x0020
        else:
                inside_code-=0xfee0
        if inside_code<0x0020 or inside_code>0x7e:      #转完之后不是半角字符返回原来的字符
                return uchar
        return chr(inside_code)
def stringQ2B(ustring):
        """把字符串全角转半角"""
        return "".join([Q2B(uchar) for uchar in ustring])
def uniform(ustring):
        """格式化字符串，完成全角转半角，大写转小写的工作"""
        return stringQ2B(ustring).lower()
def string2List(ustring):
        """将ustring按照中文，字母，数字分开"""
        retList=[]
        utmp=[]
        for uchar in ustring:
                if is_other(uchar):
                        if len(utmp)==0:
                                continue
                        else:
                                retList.append("".join(utmp))
                                utmp=[]
                else:
                        utmp.append(uchar)
        if len(utmp)!=0:
                retList.append("".join(utmp))
        return retList
if __name__=="__main__":
        print(is_alphabet("hellAword"))
        # #test Q2B and B2Q
        # for i in range(0x0020,0x007F):
        #         print Q2B(B2Q(unichr(i))),B2Q(unichr(i))
        # #test uniform
        # ustring=u'中国 人名ａ高频Ａ'
        # ustring=uniform(ustring)
        # ret=string2List(ustring)
        # print ret
