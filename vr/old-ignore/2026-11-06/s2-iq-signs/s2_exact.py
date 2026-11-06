# Exhaustive check: new gfx906 sign apply == original __vcmpne4/__vsub4 form, all grid words x all sign bytes.
import re, sys
src = open(sys.argv[1]).read()
def table(name):
    m = re.search(r'GGML_TABLE_BEGIN\((uint\d+_t),\s*' + name + r',\s*\d+\)(.*?)GGML_TABLE_END', src, re.S)
    w = int(re.search(r'\d+', m.group(1)).group()) // 8
    return w, [int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]+)', m.group(2))]

M = 0xffffffff
def vcmpne4(a):  # per byte: 0xff if byte != 0
    return sum((0xff if (a >> (8*i)) & 0xff else 0) << (8*i) for i in range(4))
def vsub4(a, b):  # per byte a - b mod 256
    return sum(((((a >> (8*i)) & 0xff) - ((b >> (8*i)) & 0xff)) & 0xff) << (8*i) for i in range(4))
def sign_mask4(s):
    bits = ((s & 0x0f) * 0x00204081) & 0x01010101
    return ((bits << 8) - bits) & M
def apply_new(g, m):
    return ((g ^ m) + (m & 0x01010101)) & M

bad = 0
# IQ2_S: sign byte b, halves from packed bits like the original
w, g2 = table('iq2s_grid')
for e in g2:
    lo, hi = e & M, e >> 32
    for b in range(256):
        s0 = vcmpne4(((b & 0x03) << 7) | ((b & 0x0C) << 21))
        s1 = vcmpne4(((b & 0x30) << 3) | ((b & 0xC0) << 17))
        bad += vsub4(lo ^ s0, s0) != apply_new(lo, sign_mask4(b))
        bad += vsub4(hi ^ s1, s1) != apply_new(hi, sign_mask4(b >> 4))
# IQ3_XXS: signs = broadcast sign byte
w, g3 = table('iq3xxs_grid')
for gx in g3:
    for gy in g3[:32]:
        for v in range(256):
            signs = v * 0x01010101
            s0 = vcmpne4(signs & 0x08040201)
            s1 = vcmpne4(signs & 0x80402010)
            bad += vsub4(gx ^ s0, s0) != apply_new(gx, sign_mask4(signs))
            bad += vsub4(gy ^ s1, s1) != apply_new(gy, sign_mask4(signs >> 4))
print("mismatches:", bad, f"(iq2s {len(g2)}x256 pairs, iq3xxs {len(g3)}x32x256)")
