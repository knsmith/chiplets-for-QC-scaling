OPENQASM 2.0;
include "qelib1.inc";
qreg q[370];
rz(pi/2) q[71];
sx q[71];
rz(-3*pi/4) q[71];
sx q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
sx q[72];
rz(-3*pi/4) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[73],q[70];
rz(-pi) q[70];
sx q[70];
cx q[70],q[71];
sx q[70];
rz(-3*pi/4) q[70];
sx q[70];
rz(pi/2) q[70];
x q[71];
cx q[71],q[72];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
sx q[73];
rz(3*pi/4) q[73];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[74];
sx q[74];
rz(-3*pi/4) q[74];
sx q[74];
rz(pi/2) q[74];
rz(pi/2) q[75];
sx q[75];
rz(-3*pi/4) q[75];
sx q[75];
rz(pi/2) q[75];
rz(pi/2) q[77];
sx q[77];
rz(-3*pi/4) q[77];
sx q[77];
rz(pi/2) q[77];
rz(pi/2) q[79];
sx q[79];
cx q[79],q[76];
rz(-pi) q[76];
sx q[76];
cx q[76],q[75];
rz(-pi) q[75];
sx q[75];
rz(pi) q[75];
cx q[75],q[74];
sx q[75];
rz(pi/2) q[75];
sx q[76];
rz(-3*pi/4) q[76];
sx q[76];
rz(pi/2) q[76];
sx q[79];
rz(-3*pi/4) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[80];
sx q[80];
rz(-3*pi/4) q[80];
sx q[80];
rz(pi/2) q[80];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[82];
sx q[82];
rz(-3*pi/4) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[83];
sx q[83];
rz(-3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
cx q[84],q[83];
x q[83];
cx q[83],q[80];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
sx q[84];
rz(-3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
sx q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
rz(pi/2) q[86];
sx q[86];
rz(-3*pi/4) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[82],q[88];
rz(-pi/2) q[82];
sx q[82];
rz(-3*pi/4) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
sx q[81];
rz(pi/2) q[81];
rz(-pi) q[88];
sx q[88];
cx q[88],q[82];
rz(-pi) q[82];
sx q[82];
rz(pi) q[82];
cx q[82],q[77];
sx q[82];
rz(pi/2) q[82];
sx q[88];
rz(-3*pi/4) q[88];
sx q[88];
rz(pi/2) q[88];
rz(pi/2) q[89];
sx q[89];
rz(-3*pi/4) q[89];
sx q[89];
rz(pi/2) q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
rz(pi/2) q[91];
sx q[91];
rz(-3*pi/4) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[92];
sx q[92];
cx q[92],q[87];
rz(-pi) q[87];
sx q[87];
cx q[87],q[86];
rz(-pi) q[86];
sx q[86];
rz(pi) q[86];
cx q[86],q[89];
sx q[86];
rz(pi/2) q[86];
sx q[87];
rz(-3*pi/4) q[87];
sx q[87];
rz(pi/2) q[87];
sx q[92];
rz(-3*pi/4) q[92];
sx q[92];
rz(pi/2) q[92];
rz(pi/2) q[93];
sx q[93];
cx q[93],q[90];
rz(-pi) q[90];
sx q[90];
rz(pi/2) q[90];
cx q[90],q[91];
sx q[90];
rz(3*pi/4) q[90];
sx q[90];
rz(pi/2) q[90];
rz(-pi) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
sx q[93];
rz(-3*pi/4) q[93];
sx q[93];
rz(pi/2) q[93];
rz(pi/2) q[94];
sx q[94];
rz(-3*pi/4) q[94];
sx q[94];
rz(pi/2) q[94];
rz(pi/2) q[95];
sx q[95];
rz(-3*pi/4) q[95];
sx q[95];
rz(pi/2) q[95];
rz(pi/2) q[97];
sx q[97];
rz(-3*pi/4) q[97];
sx q[97];
rz(pi/2) q[97];
rz(pi/2) q[98];
sx q[98];
rz(-3*pi/4) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[92],q[98];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
cx q[99],q[96];
x q[96];
cx q[96],q[95];
x q[95];
cx q[95],q[94];
rz(pi/2) q[95];
sx q[95];
rz(pi/2) q[95];
sx q[96];
rz(3*pi/4) q[96];
sx q[96];
rz(pi/2) q[96];
sx q[99];
rz(3*pi/4) q[99];
sx q[99];
rz(pi/2) q[99];
rz(pi/2) q[100];
sx q[100];
rz(-3*pi/4) q[100];
sx q[100];
rz(pi/2) q[100];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
rz(pi/2) q[102];
sx q[102];
rz(-3*pi/4) q[102];
rz(pi/2) q[103];
sx q[103];
rz(-3*pi/4) q[103];
sx q[103];
rz(pi/2) q[103];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
cx q[105],q[104];
x q[104];
cx q[104],q[103];
rz(-pi) q[103];
sx q[103];
rz(pi) q[103];
cx q[103],q[100];
sx q[103];
rz(pi/2) q[103];
rz(-pi/2) q[104];
sx q[104];
rz(-3*pi/4) q[104];
sx q[104];
rz(pi/2) q[104];
sx q[105];
rz(3*pi/4) q[105];
sx q[105];
rz(pi/2) q[105];
rz(pi/2) q[106];
sx q[106];
rz(-3*pi/4) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[101],q[102];
sx q[101];
rz(3*pi/4) q[101];
sx q[101];
rz(pi/2) q[101];
rz(-pi) q[102];
sx q[102];
rz(pi/2) q[102];
sx q[108];
rz(pi/2) q[108];
cx q[102],q[108];
sx q[102];
rz(3*pi/4) q[102];
sx q[102];
rz(-pi) q[108];
sx q[108];
rz(pi/2) q[108];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[97];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
rz(pi/2) q[108];
rz(pi/2) q[109];
sx q[109];
rz(-3*pi/4) q[109];
sx q[109];
rz(pi/2) q[109];
rz(pi/2) q[111];
sx q[111];
rz(-3*pi/4) q[111];
sx q[111];
rz(pi/2) q[111];
rz(pi/2) q[112];
sx q[112];
cx q[112],q[107];
x q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[109];
sx q[106];
rz(3*pi/4) q[106];
sx q[106];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
rz(pi/2) q[107];
rz(-pi) q[109];
sx q[109];
rz(pi) q[109];
cx q[109],q[106];
sx q[109];
rz(pi/2) q[109];
sx q[112];
rz(-3*pi/4) q[112];
sx q[112];
rz(pi/2) q[112];
rz(pi/2) q[113];
sx q[113];
rz(pi/2) q[113];
cx q[113],q[110];
rz(-pi) q[110];
sx q[110];
rz(pi/2) q[110];
cx q[110],q[111];
sx q[110];
rz(3*pi/4) q[110];
sx q[110];
rz(pi/2) q[110];
rz(-pi) q[111];
sx q[111];
rz(pi/2) q[111];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
sx q[113];
rz(3*pi/4) q[113];
sx q[113];
rz(pi/2) q[113];
rz(pi/2) q[114];
sx q[114];
rz(-3*pi/4) q[114];
sx q[114];
rz(pi/2) q[114];
rz(pi/2) q[115];
sx q[115];
rz(-3*pi/4) q[115];
sx q[115];
rz(pi/2) q[115];
rz(pi/2) q[117];
sx q[117];
rz(-3*pi/4) q[117];
sx q[117];
rz(pi/2) q[117];
rz(pi/2) q[118];
sx q[118];
rz(-3*pi/4) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[112],q[118];
rz(pi/2) q[112];
sx q[112];
rz(pi/2) q[112];
rz(pi/2) q[119];
sx q[119];
cx q[119],q[116];
x q[116];
cx q[116],q[115];
rz(-pi) q[115];
sx q[115];
rz(pi) q[115];
cx q[115],q[114];
sx q[115];
rz(pi/2) q[115];
sx q[116];
rz(3*pi/4) q[116];
sx q[116];
rz(pi/2) q[116];
sx q[119];
rz(-3*pi/4) q[119];
sx q[119];
rz(pi/2) q[119];
rz(pi/2) q[120];
sx q[120];
rz(-3*pi/4) q[120];
sx q[120];
rz(pi/2) q[120];
rz(pi/2) q[121];
sx q[121];
rz(pi/2) q[121];
rz(pi/2) q[122];
sx q[122];
rz(-3*pi/4) q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
sx q[121];
rz(pi/2) q[121];
x q[121];
rz(-pi/2) q[122];
rz(pi/2) q[123];
sx q[123];
rz(-3*pi/4) q[123];
sx q[123];
rz(pi/2) q[123];
rz(pi/2) q[125];
sx q[125];
cx q[125],q[124];
x q[124];
cx q[124],q[123];
x q[123];
cx q[123],q[120];
rz(pi/2) q[123];
sx q[123];
rz(pi/2) q[123];
sx q[124];
rz(3*pi/4) q[124];
sx q[124];
rz(pi/2) q[124];
sx q[125];
rz(-3*pi/4) q[125];
sx q[125];
rz(pi/2) q[125];
rz(pi/2) q[126];
sx q[126];
rz(-3*pi/4) q[126];
sx q[126];
rz(pi/2) q[126];
x q[128];
cx q[128],q[122];
cx q[122],q[128];
rz(-pi/2) q[122];
cx q[121],q[122];
cx q[122],q[121];
sx q[121];
rz(-3*pi/4) q[121];
sx q[121];
rz(pi/2) q[121];
cx q[122],q[117];
rz(pi/2) q[122];
sx q[122];
rz(pi/2) q[122];
sx q[128];
rz(-3*pi/4) q[128];
sx q[128];
rz(pi/2) q[128];
rz(pi/2) q[129];
sx q[129];
rz(-3*pi/4) q[129];
sx q[129];
rz(pi/2) q[129];
rz(pi/2) q[131];
sx q[131];
rz(-3*pi/4) q[131];
sx q[131];
rz(pi/2) q[131];
rz(pi/2) q[132];
sx q[132];
cx q[132],q[127];
rz(-pi) q[127];
sx q[127];
rz(pi/2) q[127];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[129];
sx q[126];
rz(3*pi/4) q[126];
sx q[126];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
rz(pi/2) q[127];
rz(-pi) q[129];
sx q[129];
rz(pi) q[129];
cx q[129],q[126];
sx q[129];
rz(pi/2) q[129];
sx q[132];
rz(-3*pi/4) q[132];
sx q[132];
rz(pi/2) q[132];
rz(pi/2) q[133];
sx q[133];
rz(pi/2) q[133];
cx q[133],q[130];
rz(-pi) q[130];
sx q[130];
rz(pi/2) q[130];
cx q[130],q[131];
sx q[130];
rz(3*pi/4) q[130];
sx q[130];
rz(pi/2) q[130];
x q[131];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
sx q[133];
rz(3*pi/4) q[133];
sx q[133];
rz(pi/2) q[133];
rz(pi/2) q[134];
sx q[134];
rz(-3*pi/4) q[134];
sx q[134];
rz(pi/2) q[134];
rz(pi/2) q[135];
sx q[135];
rz(-3*pi/4) q[135];
sx q[135];
rz(pi/2) q[135];
rz(pi/2) q[137];
sx q[137];
rz(-3*pi/4) q[137];
sx q[137];
rz(pi/2) q[137];
rz(pi/2) q[138];
sx q[138];
rz(-3*pi/4) q[138];
sx q[138];
rz(pi/2) q[138];
cx q[132],q[138];
rz(pi/2) q[132];
sx q[132];
rz(pi/2) q[132];
rz(pi/2) q[139];
sx q[139];
cx q[139],q[136];
x q[136];
cx q[136],q[135];
x q[135];
cx q[135],q[134];
rz(pi/2) q[135];
sx q[135];
rz(pi/2) q[135];
sx q[136];
rz(3*pi/4) q[136];
sx q[136];
rz(pi/2) q[136];
sx q[139];
rz(-3*pi/4) q[139];
sx q[139];
rz(pi/2) q[139];
rz(pi/2) q[140];
sx q[140];
rz(-3*pi/4) q[140];
sx q[140];
rz(pi/2) q[140];
rz(pi/2) q[141];
sx q[141];
rz(pi/2) q[141];
rz(pi/2) q[142];
sx q[142];
rz(-3*pi/4) q[142];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
sx q[141];
rz(pi/2) q[141];
rz(-pi/2) q[142];
rz(pi/2) q[143];
sx q[143];
rz(-3*pi/4) q[143];
sx q[143];
rz(pi/2) q[143];
rz(pi/2) q[145];
sx q[145];
cx q[145],q[144];
x q[144];
cx q[144],q[143];
rz(-pi) q[143];
sx q[143];
rz(pi) q[143];
cx q[143],q[140];
sx q[143];
rz(pi/2) q[143];
rz(-pi/2) q[144];
sx q[144];
rz(-3*pi/4) q[144];
sx q[144];
rz(pi/2) q[144];
sx q[145];
rz(-3*pi/4) q[145];
sx q[145];
rz(pi/2) q[145];
rz(pi/2) q[146];
sx q[146];
rz(-3*pi/4) q[146];
sx q[146];
rz(pi/2) q[146];
x q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[142],q[141];
x q[141];
sx q[142];
cx q[142],q[137];
cx q[137],q[142];
cx q[142],q[137];
rz(3*pi/4) q[137];
sx q[137];
rz(pi/2) q[137];
cx q[141],q[142];
rz(pi/2) q[141];
sx q[141];
rz(pi/2) q[141];
sx q[148];
rz(-3*pi/4) q[148];
sx q[148];
rz(pi/2) q[148];
rz(pi/2) q[149];
sx q[149];
rz(-3*pi/4) q[149];
sx q[149];
rz(pi/2) q[149];
cx q[146],q[149];
cx q[149],q[146];
cx q[146],q[149];
rz(pi/2) q[151];
sx q[151];
rz(-3*pi/4) q[151];
sx q[151];
rz(pi/2) q[151];
rz(pi/2) q[152];
sx q[152];
cx q[152],q[147];
rz(-pi) q[147];
sx q[147];
cx q[147],q[146];
rz(-pi) q[146];
sx q[146];
rz(pi) q[146];
cx q[146],q[149];
sx q[146];
rz(pi/2) q[146];
sx q[147];
rz(-3*pi/4) q[147];
sx q[147];
rz(pi/2) q[147];
sx q[152];
rz(-3*pi/4) q[152];
rz(pi/2) q[153];
sx q[153];
rz(pi/2) q[153];
cx q[153],q[150];
x q[150];
cx q[150],q[151];
sx q[150];
rz(3*pi/4) q[150];
sx q[150];
rz(pi/2) q[150];
x q[151];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
sx q[151];
rz(pi/2) q[151];
sx q[153];
rz(3*pi/4) q[153];
sx q[153];
rz(pi/2) q[153];
rz(pi/2) q[154];
sx q[154];
rz(-3*pi/4) q[154];
sx q[154];
rz(pi/2) q[154];
rz(pi/2) q[155];
sx q[155];
rz(-3*pi/4) q[155];
sx q[155];
rz(pi/2) q[155];
rz(pi/2) q[157];
sx q[157];
rz(-3*pi/4) q[157];
sx q[157];
rz(pi/2) q[157];
rz(pi/2) q[158];
sx q[158];
rz(-3*pi/4) q[158];
sx q[158];
rz(pi/2) q[158];
cx q[152],q[158];
rz(pi/2) q[152];
sx q[152];
rz(pi/2) q[152];
rz(pi/2) q[159];
sx q[159];
cx q[159],q[156];
rz(-pi) q[156];
sx q[156];
rz(pi/2) q[156];
cx q[156],q[155];
x q[155];
cx q[155],q[154];
rz(pi/2) q[155];
sx q[155];
rz(pi/2) q[155];
sx q[156];
rz(3*pi/4) q[156];
sx q[156];
rz(pi/2) q[156];
sx q[159];
rz(-3*pi/4) q[159];
sx q[159];
rz(pi/2) q[159];
rz(pi/2) q[160];
sx q[160];
rz(-3*pi/4) q[160];
sx q[160];
rz(pi/2) q[160];
rz(pi/2) q[161];
sx q[161];
rz(pi/2) q[161];
rz(pi/2) q[162];
sx q[162];
rz(-3*pi/4) q[162];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
sx q[161];
rz(pi/2) q[161];
rz(pi/2) q[163];
sx q[163];
rz(-3*pi/4) q[163];
sx q[163];
rz(pi/2) q[163];
rz(pi/2) q[165];
sx q[165];
rz(pi/2) q[165];
cx q[165],q[164];
rz(-pi) q[164];
sx q[164];
rz(pi/2) q[164];
cx q[164],q[163];
x q[163];
cx q[163],q[160];
rz(pi/2) q[163];
sx q[163];
rz(pi/2) q[163];
sx q[164];
rz(3*pi/4) q[164];
sx q[164];
rz(pi/2) q[164];
sx q[165];
rz(3*pi/4) q[165];
sx q[165];
rz(pi/2) q[165];
rz(pi/2) q[166];
sx q[166];
rz(-3*pi/4) q[166];
sx q[166];
rz(pi/2) q[166];
cx q[162],q[168];
rz(-pi/2) q[162];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
sx q[161];
rz(-3*pi/4) q[161];
sx q[161];
rz(pi/2) q[161];
x q[168];
cx q[168],q[162];
rz(-pi) q[162];
sx q[162];
rz(pi) q[162];
cx q[162],q[157];
sx q[162];
rz(pi/2) q[162];
rz(-pi/2) q[168];
sx q[168];
rz(-3*pi/4) q[168];
sx q[168];
rz(pi/2) q[168];
rz(-pi/2) q[169];
sx q[169];
rz(-pi/4) q[169];
sx q[169];
rz(-pi/2) q[169];
rz(pi/2) q[171];
sx q[171];
rz(-3*pi/4) q[171];
sx q[171];
rz(pi/2) q[171];
rz(pi/2) q[172];
sx q[172];
cx q[172],q[167];
rz(-pi) q[167];
sx q[167];
rz(pi/2) q[167];
cx q[167],q[166];
cx q[166],q[167];
cx q[167],q[166];
rz(-pi/2) q[166];
cx q[169],q[166];
cx q[166],q[169];
cx q[166],q[167];
rz(pi/2) q[166];
sx q[166];
rz(pi/2) q[166];
sx q[169];
rz(-3*pi/4) q[169];
sx q[169];
rz(pi/2) q[169];
sx q[172];
rz(-3*pi/4) q[172];
rz(pi/2) q[173];
sx q[173];
cx q[173],q[170];
x q[170];
cx q[170],q[171];
rz(-pi/2) q[170];
sx q[170];
rz(-3*pi/4) q[170];
sx q[170];
rz(pi/2) q[170];
x q[171];
sx q[173];
rz(-3*pi/4) q[173];
sx q[173];
rz(pi/2) q[173];
rz(pi/2) q[174];
sx q[174];
rz(-3*pi/4) q[174];
sx q[174];
rz(pi/2) q[174];
rz(pi/2) q[175];
sx q[175];
rz(-3*pi/4) q[175];
sx q[175];
rz(pi/2) q[175];
rz(pi/2) q[177];
sx q[177];
rz(-3*pi/4) q[177];
sx q[177];
rz(pi/2) q[177];
rz(pi/2) q[178];
sx q[178];
rz(-3*pi/4) q[178];
sx q[178];
rz(pi/2) q[178];
cx q[172],q[178];
cx q[178],q[172];
cx q[172],q[178];
cx q[171],q[172];
rz(pi/2) q[171];
sx q[171];
rz(pi/2) q[171];
sx q[178];
rz(pi/2) q[178];
rz(pi/2) q[179];
sx q[179];
cx q[179],q[176];
x q[176];
cx q[176],q[175];
x q[175];
cx q[175],q[174];
rz(pi/2) q[175];
sx q[175];
rz(pi/2) q[175];
rz(-pi/2) q[176];
sx q[176];
rz(-3*pi/4) q[176];
sx q[176];
rz(pi/2) q[176];
sx q[179];
rz(-3*pi/4) q[179];
sx q[179];
rz(pi/2) q[179];
rz(pi/2) q[180];
sx q[180];
rz(-3*pi/4) q[180];
sx q[180];
rz(pi/2) q[180];
rz(pi/2) q[181];
sx q[181];
rz(pi/2) q[181];
rz(pi/2) q[182];
sx q[182];
rz(-3*pi/4) q[182];
rz(pi/2) q[183];
sx q[183];
rz(-3*pi/4) q[183];
sx q[183];
rz(pi/2) q[183];
rz(pi/2) q[185];
sx q[185];
rz(pi/2) q[185];
cx q[185],q[184];
x q[184];
cx q[184],q[183];
rz(-pi) q[183];
sx q[183];
rz(pi) q[183];
cx q[183],q[180];
sx q[183];
rz(pi/2) q[183];
rz(-pi/2) q[184];
sx q[184];
rz(-3*pi/4) q[184];
sx q[184];
rz(pi/2) q[184];
sx q[185];
rz(3*pi/4) q[185];
sx q[185];
rz(pi/2) q[185];
rz(pi/2) q[186];
sx q[186];
rz(-3*pi/4) q[186];
sx q[186];
rz(pi/2) q[186];
cx q[182],q[188];
cx q[188],q[182];
cx q[182],q[188];
cx q[181],q[182];
sx q[181];
rz(3*pi/4) q[181];
sx q[181];
rz(pi/2) q[181];
x q[182];
sx q[188];
rz(pi/2) q[188];
cx q[182],q[188];
sx q[182];
rz(3*pi/4) q[182];
sx q[182];
cx q[182],q[177];
cx q[177],q[182];
cx q[182],q[177];
rz(pi/2) q[177];
rz(-pi) q[188];
sx q[188];
rz(pi) q[188];
cx q[188],q[182];
sx q[188];
rz(pi/2) q[188];
rz(pi/2) q[189];
sx q[189];
rz(-3*pi/4) q[189];
sx q[189];
rz(pi/2) q[189];
cx q[186],q[189];
cx q[189],q[186];
cx q[186],q[189];
rz(pi/2) q[191];
sx q[191];
rz(-3*pi/4) q[191];
sx q[191];
rz(pi/2) q[191];
rz(pi/2) q[192];
sx q[192];
rz(pi/2) q[192];
cx q[192],q[187];
rz(-pi) q[187];
sx q[187];
cx q[187],q[186];
x q[186];
cx q[186],q[189];
rz(pi/2) q[186];
sx q[186];
rz(pi/2) q[186];
sx q[187];
rz(-3*pi/4) q[187];
sx q[187];
rz(pi/2) q[187];
sx q[192];
rz(3*pi/4) q[192];
sx q[192];
rz(pi/2) q[193];
sx q[193];
rz(pi/2) q[193];
cx q[193],q[190];
x q[190];
cx q[190],q[191];
rz(-pi/2) q[190];
sx q[190];
rz(-3*pi/4) q[190];
sx q[190];
rz(pi/2) q[190];
x q[191];
sx q[193];
rz(3*pi/4) q[193];
sx q[193];
rz(pi/2) q[193];
rz(pi/2) q[194];
sx q[194];
rz(-3*pi/4) q[194];
sx q[194];
rz(pi/2) q[194];
rz(pi/2) q[195];
sx q[195];
rz(-3*pi/4) q[195];
sx q[195];
rz(pi/2) q[195];
rz(pi/2) q[197];
sx q[197];
rz(-3*pi/4) q[197];
sx q[197];
rz(pi/2) q[197];
rz(pi/2) q[198];
sx q[198];
rz(-3*pi/4) q[198];
sx q[198];
rz(pi/2) q[198];
cx q[192],q[198];
cx q[198],q[192];
cx q[192],q[198];
cx q[191],q[192];
rz(pi/2) q[191];
sx q[191];
rz(pi/2) q[191];
rz(pi/2) q[198];
rz(pi/2) q[199];
sx q[199];
rz(pi/2) q[199];
cx q[199],q[196];
x q[196];
cx q[196],q[195];
rz(-pi) q[195];
sx q[195];
rz(pi) q[195];
cx q[195],q[194];
sx q[195];
rz(pi/2) q[195];
sx q[196];
rz(3*pi/4) q[196];
sx q[196];
rz(pi/2) q[196];
sx q[199];
rz(3*pi/4) q[199];
sx q[199];
rz(pi/2) q[199];
rz(pi/2) q[200];
sx q[200];
rz(-3*pi/4) q[200];
sx q[200];
rz(pi/2) q[200];
rz(pi/2) q[201];
sx q[201];
rz(pi/2) q[201];
rz(pi/2) q[202];
sx q[202];
rz(-3*pi/4) q[202];
cx q[202],q[201];
cx q[201],q[202];
cx q[202],q[201];
sx q[201];
rz(pi/2) q[201];
rz(pi/2) q[203];
sx q[203];
rz(-3*pi/4) q[203];
sx q[203];
rz(pi/2) q[203];
rz(pi/2) q[205];
sx q[205];
rz(pi/2) q[205];
cx q[205],q[204];
rz(-pi) q[204];
sx q[204];
rz(pi/2) q[204];
cx q[204],q[203];
rz(-pi) q[203];
sx q[203];
rz(pi) q[203];
cx q[203],q[200];
sx q[203];
rz(pi/2) q[203];
sx q[204];
rz(3*pi/4) q[204];
sx q[204];
rz(pi/2) q[204];
sx q[205];
rz(3*pi/4) q[205];
sx q[205];
rz(pi/2) q[205];
rz(pi/2) q[206];
sx q[206];
rz(-3*pi/4) q[206];
sx q[206];
rz(pi/2) q[206];
cx q[202],q[208];
sx q[202];
rz(3*pi/4) q[202];
sx q[202];
cx q[202],q[201];
cx q[201],q[202];
cx q[202],q[201];
rz(pi/2) q[201];
rz(-pi) q[208];
sx q[208];
rz(pi/2) q[208];
cx q[208],q[202];
rz(-pi) q[202];
sx q[202];
rz(pi) q[202];
cx q[202],q[197];
sx q[202];
rz(pi/2) q[202];
sx q[208];
rz(3*pi/4) q[208];
sx q[208];
rz(pi/2) q[208];
rz(pi/2) q[209];
sx q[209];
rz(-3*pi/4) q[209];
sx q[209];
rz(pi/2) q[209];
cx q[206],q[209];
cx q[209],q[206];
cx q[206],q[209];
rz(pi/2) q[211];
sx q[211];
rz(-3*pi/4) q[211];
sx q[211];
rz(pi/2) q[211];
rz(pi/2) q[212];
sx q[212];
rz(pi/2) q[212];
cx q[212],q[207];
rz(-pi) q[207];
sx q[207];
cx q[207],q[206];
x q[206];
cx q[206],q[209];
rz(pi/2) q[206];
sx q[206];
rz(pi/2) q[206];
sx q[207];
rz(-3*pi/4) q[207];
sx q[207];
rz(pi/2) q[207];
sx q[212];
rz(3*pi/4) q[212];
sx q[212];
rz(pi/2) q[212];
rz(pi/2) q[213];
sx q[213];
rz(pi/2) q[213];
cx q[213],q[210];
rz(-pi) q[210];
sx q[210];
cx q[210],q[211];
sx q[210];
rz(-3*pi/4) q[210];
sx q[210];
rz(pi/2) q[210];
x q[211];
sx q[213];
rz(3*pi/4) q[213];
sx q[213];
rz(pi/2) q[213];
rz(pi/2) q[214];
sx q[214];
rz(-3*pi/4) q[214];
sx q[214];
rz(pi/2) q[214];
rz(pi/2) q[215];
sx q[215];
rz(-3*pi/4) q[215];
sx q[215];
rz(pi/2) q[215];
rz(pi/2) q[217];
sx q[217];
rz(-3*pi/4) q[217];
sx q[217];
rz(pi/2) q[217];
rz(pi/2) q[218];
sx q[218];
rz(-3*pi/4) q[218];
sx q[218];
rz(pi/2) q[218];
cx q[212],q[218];
cx q[218],q[212];
cx q[212],q[218];
cx q[211],q[212];
rz(pi/2) q[211];
sx q[211];
rz(pi/2) q[211];
rz(pi/2) q[219];
sx q[219];
cx q[219],q[216];
x q[216];
cx q[216],q[215];
x q[215];
cx q[215],q[214];
rz(pi/2) q[215];
sx q[215];
rz(pi/2) q[215];
rz(-pi/2) q[216];
sx q[216];
rz(-3*pi/4) q[216];
sx q[216];
rz(pi/2) q[216];
sx q[219];
rz(-3*pi/4) q[219];
sx q[219];
rz(pi/2) q[219];
rz(pi/2) q[220];
sx q[220];
rz(-3*pi/4) q[220];
sx q[220];
rz(pi/2) q[220];
rz(pi/2) q[221];
sx q[221];
rz(pi/2) q[221];
rz(pi/2) q[222];
sx q[222];
rz(-3*pi/4) q[222];
cx q[222],q[221];
cx q[221],q[222];
cx q[222],q[221];
sx q[221];
rz(pi/2) q[221];
rz(-pi/2) q[222];
rz(pi/2) q[223];
sx q[223];
rz(-3*pi/4) q[223];
sx q[223];
rz(pi/2) q[223];
rz(pi/2) q[225];
sx q[225];
rz(pi/2) q[225];
cx q[225],q[224];
rz(-pi) q[224];
sx q[224];
cx q[224],q[223];
rz(-pi) q[223];
sx q[223];
rz(pi) q[223];
cx q[223],q[220];
sx q[223];
rz(pi/2) q[223];
sx q[224];
rz(-3*pi/4) q[224];
sx q[224];
rz(pi/2) q[224];
sx q[225];
rz(3*pi/4) q[225];
sx q[225];
rz(pi/2) q[225];
rz(pi/2) q[226];
sx q[226];
rz(-3*pi/4) q[226];
sx q[226];
rz(pi/2) q[226];
x q[228];
cx q[228],q[222];
cx q[222],q[228];
cx q[222],q[221];
x q[221];
rz(-pi/2) q[222];
cx q[222],q[217];
cx q[217],q[222];
cx q[222],q[217];
sx q[217];
rz(-3*pi/4) q[217];
sx q[217];
rz(pi/2) q[217];
cx q[221],q[222];
rz(pi/2) q[221];
sx q[221];
rz(pi/2) q[221];
sx q[228];
rz(-3*pi/4) q[228];
sx q[228];
rz(pi/2) q[228];
rz(pi/2) q[229];
sx q[229];
rz(-3*pi/4) q[229];
sx q[229];
rz(pi/2) q[229];
cx q[226],q[229];
cx q[229],q[226];
cx q[226],q[229];
rz(pi/2) q[231];
sx q[231];
rz(-3*pi/4) q[231];
sx q[231];
rz(pi/2) q[231];
rz(pi/2) q[232];
sx q[232];
cx q[232],q[227];
rz(-pi) q[227];
sx q[227];
cx q[227],q[226];
x q[226];
cx q[226],q[229];
rz(pi/2) q[226];
sx q[226];
rz(pi/2) q[226];
sx q[227];
rz(-3*pi/4) q[227];
sx q[227];
rz(pi/2) q[227];
sx q[232];
rz(-3*pi/4) q[232];
sx q[232];
rz(pi/2) q[232];
rz(pi/2) q[233];
sx q[233];
rz(pi/2) q[233];
cx q[233],q[230];
rz(-pi) q[230];
sx q[230];
rz(pi/2) q[230];
cx q[230],q[231];
sx q[230];
rz(3*pi/4) q[230];
sx q[230];
rz(pi/2) q[230];
rz(-pi) q[231];
sx q[231];
rz(pi) q[231];
sx q[233];
rz(3*pi/4) q[233];
sx q[233];
rz(pi/2) q[233];
rz(pi/2) q[234];
sx q[234];
rz(-3*pi/4) q[234];
sx q[234];
rz(pi/2) q[234];
rz(pi/2) q[235];
sx q[235];
rz(-3*pi/4) q[235];
sx q[235];
rz(pi/2) q[235];
rz(pi/2) q[237];
sx q[237];
rz(-3*pi/4) q[237];
sx q[237];
rz(pi/2) q[237];
rz(pi/2) q[238];
sx q[238];
rz(-3*pi/4) q[238];
sx q[238];
rz(pi/2) q[238];
cx q[232],q[238];
cx q[238],q[232];
cx q[232],q[238];
cx q[231],q[232];
sx q[231];
rz(pi/2) q[231];
rz(pi/2) q[239];
sx q[239];
cx q[239],q[236];
rz(-pi) q[236];
sx q[236];
rz(pi/2) q[236];
cx q[236],q[235];
rz(-pi) q[235];
sx q[235];
rz(pi) q[235];
cx q[235],q[234];
sx q[235];
rz(pi/2) q[235];
sx q[236];
rz(3*pi/4) q[236];
sx q[236];
rz(pi/2) q[236];
sx q[239];
rz(-3*pi/4) q[239];
sx q[239];
rz(pi/2) q[239];
rz(pi/2) q[240];
sx q[240];
rz(-3*pi/4) q[240];
sx q[240];
rz(pi/2) q[240];
rz(pi/2) q[241];
sx q[241];
rz(pi/2) q[241];
rz(pi/2) q[242];
sx q[242];
rz(-3*pi/4) q[242];
cx q[242],q[241];
cx q[241],q[242];
cx q[242],q[241];
sx q[241];
rz(pi/2) q[241];
rz(pi/2) q[243];
sx q[243];
rz(-3*pi/4) q[243];
sx q[243];
rz(pi/2) q[243];
rz(pi/2) q[245];
sx q[245];
cx q[245],q[244];
rz(-pi) q[244];
sx q[244];
rz(pi/2) q[244];
cx q[244],q[243];
x q[243];
cx q[243],q[240];
rz(pi/2) q[243];
sx q[243];
rz(pi/2) q[243];
sx q[244];
rz(3*pi/4) q[244];
sx q[244];
rz(pi/2) q[244];
sx q[245];
rz(-3*pi/4) q[245];
sx q[245];
rz(pi/2) q[245];
rz(pi/2) q[246];
sx q[246];
rz(-3*pi/4) q[246];
sx q[246];
rz(pi/2) q[246];
cx q[242],q[248];
sx q[242];
rz(3*pi/4) q[242];
sx q[242];
cx q[242],q[241];
cx q[241],q[242];
cx q[242],q[241];
rz(pi/2) q[241];
rz(-pi) q[248];
sx q[248];
rz(pi/2) q[248];
cx q[248],q[242];
rz(-pi) q[242];
sx q[242];
rz(pi) q[242];
cx q[242],q[237];
sx q[242];
rz(pi/2) q[242];
sx q[248];
rz(3*pi/4) q[248];
sx q[248];
rz(pi/2) q[248];
rz(-pi/2) q[249];
sx q[249];
rz(-pi/4) q[249];
sx q[249];
rz(-pi/2) q[249];
rz(pi/2) q[251];
sx q[251];
rz(-3*pi/4) q[251];
sx q[251];
rz(pi/2) q[251];
rz(pi/2) q[252];
sx q[252];
cx q[252],q[247];
x q[247];
cx q[247],q[246];
cx q[246],q[247];
cx q[247],q[246];
rz(-pi/2) q[246];
cx q[249],q[246];
cx q[246],q[249];
cx q[246],q[247];
rz(pi/2) q[246];
sx q[246];
rz(pi/2) q[246];
sx q[249];
rz(-3*pi/4) q[249];
sx q[249];
rz(pi/2) q[249];
sx q[252];
rz(-3*pi/4) q[252];
rz(pi/2) q[253];
sx q[253];
cx q[253],q[250];
x q[250];
cx q[250],q[251];
sx q[250];
rz(3*pi/4) q[250];
sx q[250];
rz(pi/2) q[250];
x q[251];
sx q[253];
rz(-3*pi/4) q[253];
sx q[253];
rz(pi/2) q[253];
rz(pi/2) q[254];
sx q[254];
rz(-3*pi/4) q[254];
sx q[254];
rz(pi/2) q[254];
rz(pi/2) q[255];
sx q[255];
rz(-3*pi/4) q[255];
sx q[255];
rz(pi/2) q[255];
rz(pi/2) q[257];
sx q[257];
rz(-3*pi/4) q[257];
sx q[257];
rz(pi/2) q[257];
rz(pi/2) q[258];
sx q[258];
rz(-3*pi/4) q[258];
sx q[258];
rz(pi/2) q[258];
cx q[252],q[258];
cx q[258],q[252];
cx q[252],q[258];
cx q[251],q[252];
rz(pi/2) q[251];
sx q[251];
rz(pi/2) q[251];
sx q[258];
rz(pi/2) q[258];
rz(pi/2) q[259];
sx q[259];
cx q[259],q[256];
x q[256];
cx q[256],q[255];
rz(-pi) q[255];
sx q[255];
rz(pi) q[255];
cx q[255],q[254];
sx q[255];
rz(pi/2) q[255];
rz(-pi/2) q[256];
sx q[256];
rz(-3*pi/4) q[256];
sx q[256];
rz(pi/2) q[256];
sx q[259];
rz(-3*pi/4) q[259];
sx q[259];
rz(pi/2) q[259];
rz(pi/2) q[260];
sx q[260];
rz(-3*pi/4) q[260];
sx q[260];
rz(pi/2) q[260];
rz(pi/2) q[261];
sx q[261];
rz(pi/2) q[262];
sx q[262];
rz(-3*pi/4) q[262];
rz(pi/2) q[263];
sx q[263];
rz(-3*pi/4) q[263];
sx q[263];
rz(pi/2) q[263];
rz(pi/2) q[265];
sx q[265];
cx q[265],q[264];
rz(-pi) q[264];
sx q[264];
cx q[264],q[263];
rz(-pi) q[263];
sx q[263];
rz(pi) q[263];
cx q[263],q[260];
sx q[263];
rz(pi/2) q[263];
sx q[264];
rz(-3*pi/4) q[264];
sx q[264];
rz(pi/2) q[264];
sx q[265];
rz(-3*pi/4) q[265];
sx q[265];
rz(pi/2) q[265];
rz(pi/2) q[266];
sx q[266];
rz(-3*pi/4) q[266];
sx q[266];
rz(pi/2) q[266];
cx q[262],q[268];
cx q[268],q[262];
cx q[262],q[268];
cx q[261],q[262];
sx q[261];
rz(-3*pi/4) q[261];
sx q[261];
rz(pi/2) q[261];
x q[262];
rz(-pi/2) q[262];
sx q[268];
rz(pi/2) q[268];
x q[268];
cx q[268],q[262];
cx q[262],q[268];
cx q[262],q[257];
rz(pi/2) q[262];
sx q[262];
rz(pi/2) q[262];
sx q[268];
rz(-3*pi/4) q[268];
sx q[268];
rz(pi/2) q[268];
rz(pi/2) q[269];
sx q[269];
rz(-3*pi/4) q[269];
sx q[269];
rz(pi/2) q[269];
rz(pi/2) q[271];
sx q[271];
rz(-3*pi/4) q[271];
sx q[271];
rz(pi/2) q[271];
rz(pi/2) q[272];
sx q[272];
rz(pi/2) q[272];
cx q[272],q[267];
rz(-pi) q[267];
sx q[267];
rz(pi/2) q[267];
cx q[267],q[266];
cx q[266],q[267];
cx q[267],q[266];
cx q[266],q[269];
rz(-pi/2) q[266];
sx q[266];
rz(-3*pi/4) q[266];
cx q[267],q[266];
cx q[266],q[267];
cx q[267],q[266];
sx q[267];
rz(pi/2) q[267];
rz(-pi) q[269];
sx q[269];
rz(pi) q[269];
cx q[269],q[266];
sx q[269];
rz(pi/2) q[269];
sx q[272];
rz(3*pi/4) q[272];
sx q[272];
rz(pi/2) q[272];
rz(pi/2) q[273];
sx q[273];
cx q[273],q[270];
rz(-pi) q[270];
sx q[270];
rz(pi/2) q[270];
cx q[270],q[271];
sx q[270];
rz(3*pi/4) q[270];
sx q[270];
rz(pi/2) q[270];
rz(-pi) q[271];
sx q[271];
rz(pi/2) q[271];
cx q[272],q[271];
cx q[271],q[272];
cx q[272],q[271];
sx q[273];
rz(-3*pi/4) q[273];
sx q[273];
rz(pi/2) q[273];
rz(pi/2) q[274];
sx q[274];
rz(-3*pi/4) q[274];
sx q[274];
rz(pi/2) q[274];
rz(pi/2) q[275];
sx q[275];
rz(-3*pi/4) q[275];
sx q[275];
rz(pi/2) q[275];
rz(pi/2) q[277];
sx q[277];
rz(-3*pi/4) q[277];
sx q[277];
rz(pi/2) q[277];
rz(pi/2) q[278];
sx q[278];
rz(-3*pi/4) q[278];
sx q[278];
rz(pi/2) q[278];
cx q[272],q[278];
rz(pi/2) q[272];
sx q[272];
rz(pi/2) q[272];
rz(pi/2) q[279];
sx q[279];
cx q[279],q[276];
x q[276];
cx q[276],q[275];
x q[275];
cx q[275],q[274];
rz(pi/2) q[275];
sx q[275];
rz(pi/2) q[275];
sx q[276];
rz(3*pi/4) q[276];
sx q[276];
rz(pi/2) q[276];
sx q[279];
rz(-3*pi/4) q[279];
sx q[279];
rz(pi/2) q[279];
rz(pi/2) q[280];
sx q[280];
rz(-3*pi/4) q[280];
sx q[280];
rz(pi/2) q[280];
rz(pi/2) q[281];
sx q[281];
rz(pi/2) q[282];
sx q[282];
rz(-3*pi/4) q[282];
rz(pi/2) q[283];
sx q[283];
rz(-3*pi/4) q[283];
sx q[283];
rz(pi/2) q[283];
rz(pi/2) q[285];
sx q[285];
rz(pi/2) q[285];
cx q[285],q[284];
rz(-pi) q[284];
sx q[284];
cx q[284],q[283];
x q[283];
cx q[283],q[280];
rz(pi/2) q[283];
sx q[283];
rz(pi/2) q[283];
sx q[284];
rz(-3*pi/4) q[284];
sx q[284];
rz(pi/2) q[284];
sx q[285];
rz(3*pi/4) q[285];
sx q[285];
rz(pi/2) q[285];
rz(pi/2) q[286];
sx q[286];
rz(-3*pi/4) q[286];
sx q[286];
rz(pi/2) q[286];
cx q[282],q[288];
cx q[288],q[282];
cx q[282],q[288];
cx q[281],q[282];
sx q[281];
rz(-3*pi/4) q[281];
sx q[281];
rz(pi/2) q[281];
rz(-pi) q[282];
sx q[282];
sx q[288];
rz(pi/2) q[288];
cx q[282],q[288];
cx q[282],q[277];
cx q[277],q[282];
cx q[282],q[277];
sx q[277];
rz(-3*pi/4) q[277];
sx q[277];
rz(pi/2) q[277];
x q[288];
cx q[288],q[282];
rz(pi/2) q[288];
sx q[288];
rz(pi/2) q[288];
rz(pi/2) q[289];
sx q[289];
rz(-3*pi/4) q[289];
sx q[289];
rz(pi/2) q[289];
rz(pi/2) q[291];
sx q[291];
rz(-3*pi/4) q[291];
sx q[291];
rz(pi/2) q[291];
rz(pi/2) q[292];
sx q[292];
cx q[292],q[287];
rz(-pi) q[287];
sx q[287];
rz(pi/2) q[287];
cx q[287],q[286];
cx q[286],q[287];
cx q[287],q[286];
cx q[286],q[289];
sx q[286];
rz(3*pi/4) q[286];
sx q[286];
rz(-pi) q[289];
sx q[289];
rz(pi/2) q[289];
cx q[286],q[289];
cx q[289],q[286];
cx q[286],q[289];
cx q[286],q[287];
rz(pi/2) q[286];
sx q[286];
rz(pi/2) q[286];
rz(pi/2) q[289];
sx q[292];
rz(-3*pi/4) q[292];
rz(pi/2) q[293];
sx q[293];
cx q[293],q[290];
x q[290];
cx q[290],q[291];
sx q[290];
rz(3*pi/4) q[290];
sx q[290];
rz(pi/2) q[290];
x q[291];
cx q[292],q[291];
cx q[291],q[292];
cx q[292],q[291];
sx q[291];
rz(pi/2) q[291];
sx q[293];
rz(-3*pi/4) q[293];
sx q[293];
rz(pi/2) q[293];
rz(pi/2) q[294];
sx q[294];
rz(-3*pi/4) q[294];
sx q[294];
rz(pi/2) q[294];
rz(pi/2) q[295];
sx q[295];
rz(-3*pi/4) q[295];
sx q[295];
rz(pi/2) q[295];
rz(pi/2) q[297];
sx q[297];
rz(-3*pi/4) q[297];
sx q[297];
rz(pi/2) q[297];
rz(pi/2) q[298];
sx q[298];
rz(-3*pi/4) q[298];
sx q[298];
rz(pi/2) q[298];
cx q[292],q[298];
rz(pi/2) q[292];
sx q[292];
rz(pi/2) q[292];
rz(pi/2) q[299];
sx q[299];
cx q[299],q[296];
x q[296];
cx q[296],q[295];
rz(-pi) q[295];
sx q[295];
rz(pi) q[295];
cx q[295],q[294];
sx q[295];
rz(pi/2) q[295];
rz(-pi/2) q[296];
sx q[296];
rz(-3*pi/4) q[296];
sx q[296];
rz(pi/2) q[296];
sx q[299];
rz(-3*pi/4) q[299];
sx q[299];
rz(pi/2) q[299];
rz(pi/2) q[300];
sx q[300];
rz(-3*pi/4) q[300];
sx q[300];
rz(pi/2) q[300];
rz(pi/2) q[301];
sx q[301];
rz(pi/2) q[302];
sx q[302];
rz(-3*pi/4) q[302];
rz(pi/2) q[303];
sx q[303];
rz(-3*pi/4) q[303];
sx q[303];
rz(pi/2) q[303];
rz(pi/2) q[305];
sx q[305];
rz(pi/2) q[305];
cx q[305],q[304];
rz(-pi) q[304];
sx q[304];
cx q[304],q[303];
rz(-pi) q[303];
sx q[303];
rz(pi) q[303];
cx q[303],q[300];
sx q[303];
rz(pi/2) q[303];
sx q[304];
rz(-3*pi/4) q[304];
sx q[304];
rz(pi/2) q[304];
sx q[305];
rz(3*pi/4) q[305];
sx q[305];
rz(pi/2) q[305];
rz(pi/2) q[306];
sx q[306];
rz(-3*pi/4) q[306];
sx q[306];
rz(pi/2) q[306];
cx q[302],q[308];
cx q[308],q[302];
cx q[302],q[308];
cx q[301],q[302];
sx q[301];
rz(-3*pi/4) q[301];
sx q[301];
rz(pi/2) q[301];
rz(-pi) q[302];
sx q[302];
rz(pi/2) q[302];
sx q[308];
rz(pi/2) q[308];
cx q[302],q[308];
sx q[302];
x q[308];
cx q[302],q[308];
cx q[308],q[302];
cx q[302],q[308];
cx q[302],q[297];
rz(pi/2) q[302];
sx q[302];
rz(pi/2) q[302];
rz(3*pi/4) q[308];
sx q[308];
rz(pi/2) q[308];
rz(pi/2) q[309];
sx q[309];
rz(-3*pi/4) q[309];
sx q[309];
rz(pi/2) q[309];
cx q[306],q[309];
cx q[309],q[306];
cx q[306],q[309];
rz(pi/2) q[311];
sx q[311];
rz(-3*pi/4) q[311];
sx q[311];
rz(pi/2) q[311];
rz(pi/2) q[312];
sx q[312];
rz(pi/2) q[312];
cx q[312],q[307];
rz(-pi) q[307];
sx q[307];
cx q[307],q[306];
x q[306];
cx q[306],q[309];
rz(pi/2) q[306];
sx q[306];
rz(pi/2) q[306];
sx q[307];
rz(-3*pi/4) q[307];
sx q[307];
rz(pi/2) q[307];
sx q[312];
rz(3*pi/4) q[312];
sx q[312];
rz(pi/2) q[312];
rz(pi/2) q[313];
sx q[313];
rz(pi/2) q[313];
cx q[313],q[310];
x q[310];
cx q[310],q[311];
rz(-pi/2) q[310];
sx q[310];
rz(-3*pi/4) q[310];
sx q[310];
rz(pi/2) q[310];
rz(-pi) q[311];
sx q[311];
rz(pi/2) q[311];
cx q[312],q[311];
cx q[311],q[312];
cx q[312],q[311];
sx q[313];
rz(3*pi/4) q[313];
sx q[313];
rz(pi/2) q[313];
rz(pi/2) q[314];
sx q[314];
rz(-3*pi/4) q[314];
sx q[314];
rz(pi/2) q[314];
rz(pi/2) q[315];
sx q[315];
rz(-3*pi/4) q[315];
sx q[315];
rz(pi/2) q[315];
rz(pi/2) q[317];
sx q[317];
rz(-3*pi/4) q[317];
sx q[317];
rz(pi/2) q[317];
rz(pi/2) q[318];
sx q[318];
rz(-3*pi/4) q[318];
sx q[318];
rz(pi/2) q[318];
cx q[312],q[318];
rz(pi/2) q[312];
sx q[312];
rz(pi/2) q[312];
rz(pi/2) q[319];
sx q[319];
rz(pi/2) q[319];
cx q[319],q[316];
rz(-pi) q[316];
sx q[316];
rz(pi/2) q[316];
cx q[316],q[315];
x q[315];
cx q[315],q[314];
rz(pi/2) q[315];
sx q[315];
rz(pi/2) q[315];
sx q[316];
rz(3*pi/4) q[316];
sx q[316];
rz(pi/2) q[316];
sx q[319];
rz(3*pi/4) q[319];
sx q[319];
rz(pi/2) q[319];
rz(pi/2) q[320];
sx q[320];
rz(-3*pi/4) q[320];
sx q[320];
rz(pi/2) q[320];
rz(pi/2) q[321];
sx q[321];
rz(pi/2) q[321];
rz(pi/2) q[322];
sx q[322];
rz(-3*pi/4) q[322];
cx q[322],q[321];
cx q[321],q[322];
cx q[322],q[321];
sx q[321];
rz(pi/2) q[321];
rz(pi/2) q[323];
sx q[323];
rz(-3*pi/4) q[323];
sx q[323];
rz(pi/2) q[323];
rz(pi/2) q[325];
sx q[325];
cx q[325],q[324];
x q[324];
cx q[324],q[323];
x q[323];
cx q[323],q[320];
rz(pi/2) q[323];
sx q[323];
rz(pi/2) q[323];
sx q[324];
rz(3*pi/4) q[324];
sx q[324];
rz(pi/2) q[324];
sx q[325];
rz(-3*pi/4) q[325];
sx q[325];
rz(pi/2) q[325];
rz(pi/2) q[326];
sx q[326];
rz(-3*pi/4) q[326];
sx q[326];
rz(pi/2) q[326];
cx q[322],q[328];
rz(-pi/2) q[322];
sx q[322];
rz(-3*pi/4) q[322];
rz(-pi) q[328];
sx q[328];
rz(pi/2) q[328];
cx q[322],q[328];
cx q[328],q[322];
cx q[322],q[328];
cx q[322],q[321];
x q[321];
sx q[322];
cx q[322],q[317];
cx q[317],q[322];
cx q[322],q[317];
rz(3*pi/4) q[317];
sx q[317];
rz(pi/2) q[317];
cx q[321],q[322];
rz(pi/2) q[321];
sx q[321];
rz(pi/2) q[321];
sx q[328];
rz(pi/2) q[328];
rz(pi/2) q[329];
sx q[329];
rz(-3*pi/4) q[329];
sx q[329];
rz(pi/2) q[329];
rz(pi/2) q[331];
sx q[331];
rz(-3*pi/4) q[331];
sx q[331];
rz(pi/2) q[331];
rz(pi/2) q[332];
sx q[332];
cx q[332],q[327];
rz(-pi) q[327];
sx q[327];
rz(pi/2) q[327];
cx q[327],q[326];
cx q[326],q[327];
cx q[327],q[326];
cx q[326],q[329];
sx q[326];
cx q[327],q[326];
cx q[326],q[327];
cx q[327],q[326];
rz(3*pi/4) q[327];
sx q[327];
rz(pi/2) q[327];
x q[329];
cx q[329],q[326];
rz(pi/2) q[329];
sx q[329];
rz(pi/2) q[329];
sx q[332];
rz(-3*pi/4) q[332];
sx q[332];
rz(pi/2) q[332];
rz(pi/2) q[333];
sx q[333];
rz(pi/2) q[333];
cx q[333],q[330];
x q[330];
cx q[330],q[331];
sx q[330];
rz(3*pi/4) q[330];
sx q[330];
rz(pi/2) q[330];
rz(-pi) q[331];
sx q[331];
rz(pi/2) q[331];
cx q[332],q[331];
cx q[331],q[332];
cx q[332],q[331];
sx q[333];
rz(3*pi/4) q[333];
sx q[333];
rz(pi/2) q[333];
rz(pi/2) q[334];
sx q[334];
rz(-3*pi/4) q[334];
sx q[334];
rz(pi/2) q[334];
rz(pi/2) q[335];
sx q[335];
rz(-3*pi/4) q[335];
sx q[335];
rz(pi/2) q[335];
rz(pi/2) q[337];
sx q[337];
rz(-3*pi/4) q[337];
sx q[337];
rz(pi/2) q[337];
rz(pi/2) q[338];
sx q[338];
rz(-3*pi/4) q[338];
sx q[338];
rz(pi/2) q[338];
cx q[332],q[338];
rz(pi/2) q[332];
sx q[332];
rz(pi/2) q[332];
rz(pi/2) q[339];
sx q[339];
cx q[339],q[336];
x q[336];
cx q[336],q[335];
rz(-pi) q[335];
sx q[335];
rz(pi) q[335];
cx q[335],q[334];
sx q[335];
rz(pi/2) q[335];
sx q[336];
rz(3*pi/4) q[336];
sx q[336];
rz(pi/2) q[336];
sx q[339];
rz(-3*pi/4) q[339];
sx q[339];
rz(pi/2) q[339];
rz(pi/2) q[340];
sx q[340];
rz(-3*pi/4) q[340];
sx q[340];
rz(pi/2) q[340];
rz(pi/2) q[341];
sx q[341];
rz(pi/2) q[341];
rz(pi/2) q[342];
sx q[342];
rz(-3*pi/4) q[342];
rz(pi/2) q[343];
sx q[343];
rz(-3*pi/4) q[343];
sx q[343];
rz(pi/2) q[343];
rz(pi/2) q[345];
sx q[345];
cx q[345],q[344];
x q[344];
cx q[344],q[343];
x q[343];
cx q[343],q[340];
rz(pi/2) q[343];
sx q[343];
rz(pi/2) q[343];
rz(-pi/2) q[344];
sx q[344];
rz(-3*pi/4) q[344];
sx q[344];
rz(pi/2) q[344];
sx q[345];
rz(-3*pi/4) q[345];
sx q[345];
rz(pi/2) q[345];
rz(pi/2) q[346];
sx q[346];
rz(-3*pi/4) q[346];
sx q[346];
rz(pi/2) q[346];
cx q[342],q[348];
cx q[348],q[342];
cx q[342],q[348];
cx q[341],q[342];
sx q[341];
rz(3*pi/4) q[341];
sx q[341];
rz(pi/2) q[341];
rz(-pi) q[342];
sx q[342];
rz(pi/2) q[342];
sx q[348];
rz(pi/2) q[348];
cx q[342],q[348];
sx q[342];
cx q[342],q[337];
cx q[337],q[342];
cx q[342],q[337];
rz(3*pi/4) q[337];
sx q[337];
rz(pi/2) q[337];
x q[348];
cx q[348],q[342];
rz(pi/2) q[348];
sx q[348];
rz(pi/2) q[348];
rz(pi/2) q[349];
sx q[349];
rz(-3*pi/4) q[349];
sx q[349];
rz(pi/2) q[349];
rz(pi/2) q[351];
sx q[351];
rz(-3*pi/4) q[351];
sx q[351];
rz(pi/2) q[351];
rz(pi/2) q[352];
sx q[352];
cx q[352],q[347];
x q[347];
cx q[347],q[346];
cx q[346],q[347];
cx q[347],q[346];
cx q[346],q[349];
sx q[346];
rz(3*pi/4) q[346];
sx q[346];
rz(-pi) q[349];
sx q[349];
rz(pi/2) q[349];
cx q[346],q[349];
cx q[349],q[346];
cx q[346],q[349];
cx q[346],q[347];
rz(pi/2) q[346];
sx q[346];
rz(pi/2) q[346];
rz(pi/2) q[349];
sx q[352];
rz(-3*pi/4) q[352];
rz(pi/2) q[353];
sx q[353];
rz(pi/2) q[353];
cx q[353],q[350];
x q[350];
cx q[350],q[351];
rz(-pi/2) q[350];
sx q[350];
rz(-3*pi/4) q[350];
sx q[350];
rz(pi/2) q[350];
x q[351];
sx q[353];
rz(3*pi/4) q[353];
sx q[353];
rz(pi/2) q[353];
rz(pi/2) q[354];
sx q[354];
rz(-3*pi/4) q[354];
sx q[354];
rz(pi/2) q[354];
rz(pi/2) q[355];
sx q[355];
rz(-3*pi/4) q[355];
sx q[355];
rz(pi/2) q[355];
rz(pi/2) q[357];
sx q[357];
rz(-3*pi/4) q[357];
sx q[357];
rz(pi/2) q[357];
rz(pi/2) q[358];
sx q[358];
rz(-3*pi/4) q[358];
sx q[358];
rz(pi/2) q[358];
cx q[352],q[358];
cx q[358],q[352];
cx q[352],q[358];
cx q[351],q[352];
rz(pi/2) q[351];
sx q[351];
rz(pi/2) q[351];
sx q[358];
rz(pi/2) q[358];
rz(pi/2) q[359];
sx q[359];
rz(pi/2) q[359];
cx q[359],q[356];
x q[356];
cx q[356],q[355];
rz(-pi) q[355];
sx q[355];
rz(pi) q[355];
cx q[355],q[354];
sx q[355];
rz(pi/2) q[355];
rz(-pi/2) q[356];
sx q[356];
rz(-3*pi/4) q[356];
sx q[356];
rz(pi/2) q[356];
sx q[359];
rz(3*pi/4) q[359];
sx q[359];
rz(pi/2) q[359];
rz(pi/2) q[360];
sx q[360];
rz(-3*pi/4) q[360];
sx q[360];
rz(pi/2) q[360];
rz(pi/2) q[361];
sx q[361];
rz(pi/2) q[362];
sx q[362];
rz(-3*pi/4) q[362];
rz(pi/2) q[363];
sx q[363];
rz(-3*pi/4) q[363];
sx q[363];
rz(pi/2) q[363];
rz(pi/2) q[365];
sx q[365];
rz(pi/2) q[365];
cx q[365],q[364];
rz(-pi) q[364];
sx q[364];
rz(pi/2) q[364];
cx q[364],q[363];
rz(-pi) q[363];
sx q[363];
rz(pi) q[363];
cx q[363],q[360];
sx q[363];
rz(pi/2) q[363];
sx q[364];
rz(3*pi/4) q[364];
sx q[364];
rz(pi/2) q[364];
sx q[365];
rz(3*pi/4) q[365];
sx q[365];
rz(pi/2) q[365];
cx q[362],q[368];
cx q[368],q[362];
cx q[362],q[368];
cx q[361],q[362];
sx q[361];
rz(-3*pi/4) q[361];
sx q[361];
rz(pi/2) q[361];
x q[362];
sx q[368];
rz(pi/2) q[368];
cx q[362],q[368];
rz(-pi/2) q[362];
sx q[362];
rz(-3*pi/4) q[362];
rz(-pi) q[368];
sx q[368];
rz(pi/2) q[368];
cx q[362],q[368];
cx q[368],q[362];
cx q[362],q[368];
cx q[362],q[357];
rz(pi/2) q[362];
sx q[362];
rz(pi/2) q[362];
sx q[368];
rz(pi/2) q[368];
