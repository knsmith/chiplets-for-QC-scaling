OPENQASM 2.0;
include "qelib1.inc";
qreg q[180];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[0];
cx q[0],q[11];
cx q[11],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[0];
cx q[0],q[11];
cx q[11],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[12],q[4];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[4],q[12];
cx q[12],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[0];
cx q[0],q[11];
cx q[11],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[0];
cx q[0],q[11];
cx q[11],q[0];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[4];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[39];
cx q[39],q[42];
cx q[42],q[39];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[39];
cx q[39],q[42];
cx q[42],q[39];
cx q[39],q[28];
cx q[28],q[39];
cx q[39],q[28];
cx q[53],q[44];
cx q[44],q[53];
cx q[53],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[67],q[56];
cx q[56],q[67];
cx q[67],q[56];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[53];
cx q[53],q[58];
cx q[58],q[53];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[83],q[80];
cx q[80],q[83];
cx q[83],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[66],q[88];
cx q[88],q[66];
cx q[66],q[88];
cx q[80],q[89];
cx q[89],q[80];
cx q[80],q[89];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[80],q[89];
cx q[89],q[80];
cx q[80],q[89];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[119],q[118];
cx q[118],q[119];
cx q[119],q[118];
cx q[120],q[119];
cx q[119],q[120];
cx q[120],q[119];
cx q[119],q[118];
cx q[118],q[119];
cx q[119],q[118];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
rz(pi/2) q[138];
sx q[138];
rz(pi/2) q[138];
cx q[138],q[137];
cx q[137],q[138];
cx q[138],q[137];
cx q[137],q[136];
cx q[136],q[137];
cx q[137],q[136];
cx q[135],q[136];
cx q[136],q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[134],q[133];
cx q[133],q[134];
cx q[134],q[133];
cx q[133],q[132];
cx q[132],q[133];
cx q[133],q[132];
cx q[132],q[129];
cx q[129],q[132];
cx q[132],q[129];
cx q[137],q[136];
cx q[136],q[137];
cx q[137],q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[139],q[138];
cx q[138],q[139];
cx q[137],q[138];
cx q[138],q[139];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[139],q[140];
cx q[138],q[139];
cx q[139],q[138];
cx q[140],q[139];
cx q[139],q[138];
cx q[138],q[139];
cx q[139],q[138];
cx q[138],q[137];
cx q[137],q[138];
cx q[138],q[137];
cx q[137],q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[134],q[135];
cx q[135],q[134];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[137],q[136];
cx q[136],q[137];
cx q[137],q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[140],q[139];
cx q[139],q[140];
cx q[140],q[139];
cx q[139],q[138];
cx q[138],q[139];
cx q[139],q[138];
cx q[137],q[138];
cx q[138],q[137];
cx q[137],q[138];
cx q[138],q[137];
cx q[137],q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[133],q[134];
cx q[134],q[133];
cx q[133],q[132];
cx q[132],q[133];
cx q[133],q[132];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[140],q[131];
cx q[131],q[140];
cx q[140],q[131];
cx q[140],q[139];
cx q[139],q[140];
cx q[140],q[139];
cx q[139],q[138];
cx q[138],q[139];
cx q[139],q[138];
cx q[138],q[137];
cx q[137],q[138];
cx q[138],q[137];
cx q[136],q[137];
cx q[136],q[130];
cx q[130],q[136];
cx q[136],q[130];
cx q[137],q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[133],q[134];
cx q[134],q[133];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[136],q[130];
cx q[130],q[136];
cx q[136],q[130];
cx q[130],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[123],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[100],q[174];
cx q[174],q[100];
cx q[100],q[174];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[114],q[175];
cx q[175],q[114];
cx q[114],q[175];
cx q[128],q[176];
cx q[176],q[128];
cx q[128],q[176];
cx q[128],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[126],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[123],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[119],q[120];
cx q[120],q[119];
cx q[119],q[118];
cx q[118],q[119];
cx q[119],q[118];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[119];
cx q[119],q[120];
cx q[120],q[119];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[115];
cx q[115],q[120];
cx q[120],q[115];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[128],q[176];
cx q[176],q[128];
cx q[128],q[176];
cx q[128],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[124],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[128],q[117];
cx q[117],q[128];
cx q[128],q[117];
cx q[117],q[114];
cx q[114],q[117];
cx q[117],q[114];
cx q[114],q[175];
cx q[128],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[124],q[125];
cx q[124],q[116];
cx q[116],q[124];
cx q[124],q[116];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[128],q[117];
cx q[117],q[128];
cx q[128],q[117];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[128];
cx q[128],q[127];
cx q[127],q[126];
cx q[126],q[127];
cx q[127],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[122],q[123];
cx q[123],q[122];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[124],q[116];
cx q[116],q[124];
cx q[124],q[116];
cx q[116],q[110];
cx q[110],q[116];
cx q[116],q[110];
cx q[175],q[114];
cx q[114],q[175];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
cx q[113],q[112];
cx q[112],q[113];
cx q[113],q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[110],q[111];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
cx q[113],q[112];
cx q[112],q[113];
cx q[113],q[112];
cx q[116],q[110];
cx q[110],q[116];
cx q[116],q[110];
cx q[111],q[110];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[108],q[109];
cx q[109],q[108];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[110],q[109];
cx q[109],q[110];
cx q[112],q[111];
cx q[111],q[112];
cx q[110],q[111];
cx q[109],q[110];
cx q[110],q[109];
cx q[111],q[112];
cx q[112],q[111];
cx q[112],q[113];
cx q[113],q[112];
cx q[112],q[113];
cx q[113],q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[109],q[108];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[101];
cx q[101],q[104];
cx q[104],q[101];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[109],q[108];
cx q[108],q[109];
cx q[109],q[108];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
cx q[113],q[112];
cx q[112],q[113];
cx q[113],q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[109],q[108];
cx q[108],q[109];
cx q[109],q[108];
cx q[108],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[112],q[103];
cx q[103],q[112];
cx q[112],q[103];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[108],q[109];
cx q[108],q[102];
cx q[102],q[108];
cx q[108],q[102];
cx q[109],q[108];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[105],q[106];
cx q[106],q[105];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[108],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[108],q[102];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[92],q[93];
cx q[93],q[92];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[100],q[174];
cx q[174],q[100];
cx q[100],q[174];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[100],q[99];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[91],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[99],q[100];
cx q[100],q[99];
cx q[100],q[83];
cx q[83],q[100];
cx q[100],q[83];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[100],q[99];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[95],q[96];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[92],q[81];
cx q[81],q[92];
cx q[92],q[81];
cx q[81],q[72];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[67],q[56];
cx q[56],q[67];
cx q[67],q[56];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[67],q[56];
cx q[56],q[67];
cx q[67],q[56];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[76],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[74],q[75];
cx q[75],q[74];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[69];
cx q[69],q[78];
cx q[80],q[89];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[82],q[76];
cx q[76],q[82];
cx q[75],q[76];
cx q[76],q[82];
cx q[82],q[76];
cx q[89],q[80];
cx q[80],q[89];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[96],q[82];
cx q[82],q[96];
cx q[96],q[82];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[81],q[92];
cx q[92],q[81];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[82];
cx q[82],q[96];
cx q[96],q[82];
cx q[82],q[76];
cx q[76],q[82];
cx q[82],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[82],q[76];
cx q[76],q[82];
cx q[82],q[76];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[68];
cx q[68],q[74];
cx q[74],q[68];
cx q[68],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[69];
cx q[69],q[78];
cx q[78],q[69];
cx q[69],q[64];
cx q[64],q[69];
cx q[69],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[67];
cx q[67],q[70];
cx q[70],q[67];
cx q[67],q[56];
cx q[56],q[67];
cx q[67],q[56];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[58];
cx q[56],q[57];
cx q[57],q[56];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[67],q[56];
cx q[56],q[67];
cx q[67],q[56];
cx q[68],q[60];
cx q[60],q[68];
cx q[68],q[60];
cx q[74],q[68];
cx q[68],q[74];
cx q[74],q[68];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[68];
cx q[68],q[74];
cx q[74],q[68];
cx q[68],q[60];
cx q[60],q[68];
cx q[68],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[56],q[57];
cx q[57],q[56];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[58],q[53];
cx q[53],q[58];
cx q[58],q[53];
cx q[53],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[47],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[66],q[88];
cx q[68],q[60];
cx q[60],q[68];
cx q[68],q[60];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[68],q[74];
cx q[74],q[68];
cx q[68],q[74];
cx q[74],q[68];
cx q[68],q[60];
cx q[60],q[68];
cx q[68],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[53];
cx q[53],q[58];
cx q[58],q[53];
cx q[53],q[44];
cx q[44],q[53];
cx q[53],q[44];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[40];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[28],q[29];
cx q[29],q[28];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[39],q[28];
cx q[28],q[39];
cx q[39],q[28];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[58],q[53];
cx q[53],q[58];
cx q[58],q[53];
cx q[44],q[53];
cx q[53],q[44];
cx q[44],q[53];
cx q[53],q[44];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[0];
cx q[0],q[11];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[25],q[30];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[47],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[16];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[30],q[25];
cx q[25],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[88],q[66];
cx q[66],q[88];
cx q[66],q[55];
cx q[55],q[66];
cx q[66],q[55];
cx q[55],q[52];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[66],q[88];
cx q[88],q[66];
cx q[66],q[88];
cx q[66],q[55];
cx q[55],q[66];
cx q[66],q[55];
cx q[55],q[52];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[49],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[11];
cx q[11],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[16],q[25];
cx q[25],q[30];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[30],q[25];
cx q[30],q[31];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[25],q[16];
cx q[16],q[25];
cx q[31],q[32];
cx q[32],q[31];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[12],q[4];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[4],q[12];
cx q[12],q[4];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[52],q[87];
cx q[66],q[88];
cx q[87],q[52];
cx q[52],q[87];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[88],q[66];
cx q[66],q[88];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[48],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[4],q[12];
cx q[12],q[4];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[18],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[30],q[25];
cx q[25],q[30];
cx q[30],q[25];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[4];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[12],q[4];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[4],q[12];
cx q[12],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[88];
cx q[88],q[66];
cx q[66],q[88];
cx q[66],q[55];
cx q[55],q[66];
cx q[66],q[55];
cx q[55],q[52];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[35],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[54];
cx q[54],q[62];
cx q[62],q[54];
cx q[54],q[48];
cx q[48],q[54];
cx q[54],q[48];
cx q[66],q[55];
cx q[55],q[66];
cx q[66],q[55];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[12],q[18];
cx q[18],q[12];
cx q[18],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[16];
cx q[25],q[16];
cx q[16],q[25];
cx q[25],q[16];
cx q[25],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[40],q[32];
cx q[32],q[40];
cx q[40],q[32];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[47],q[48];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[66],q[55];
cx q[55],q[66];
cx q[66],q[55];
cx q[55],q[52];
cx q[52],q[55];
cx q[55],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[27],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[13],q[8];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[52],q[87];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[4];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[20],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[87],q[52];
cx q[52],q[87];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[35],q[34];
cx q[34],q[35];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[50],q[41];
cx q[41],q[50];
cx q[52],q[87];
cx q[87],q[52];
cx q[52],q[87];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[22],q[21];
cx q[21],q[22];
cx q[20],q[21];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[21],q[22];
cx q[22],q[21];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[22];
cx q[22],q[13];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[26],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[36],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[49],q[50];
cx q[50],q[49];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[27],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[50],q[41];
cx q[41],q[50];
cx q[50],q[41];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[41],q[36];
cx q[36],q[41];
cx q[41],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[13],q[8];
cx q[22],q[21];
cx q[21],q[22];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[22];
cx q[22],q[13];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[27],q[24];
cx q[24],q[27];
cx q[27],q[24];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[37],q[38];
cx q[38],q[37];
cx q[86],q[38];
cx q[38],q[86];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[27],q[24];
cx q[24],q[27];
cx q[27],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[27],q[24];
cx q[24],q[27];
cx q[27],q[24];
cx q[24],q[85];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[85],q[24];
cx q[24],q[85];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[13];
cx q[13],q[22];
cx q[22],q[13];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[26],q[20];
cx q[20],q[26];
cx q[26],q[20];
cx q[34],q[26];
cx q[26],q[34];
cx q[34],q[26];
cx q[34],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[37],q[38];
cx q[38],q[27];
cx q[27],q[38];
cx q[38],q[27];
cx q[27],q[24];