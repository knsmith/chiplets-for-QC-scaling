OPENQASM 2.0;
include "qelib1.inc";
qreg q[220];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[22],q[28];
cx q[28],q[22];
cx q[22],q[28];
rz(pi/2) q[32];
sx q[32];
rz(pi/2) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[28],q[30];
cx q[30],q[28];
cx q[22],q[28];
cx q[28],q[22];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[28];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[28],q[22];
cx q[22],q[28];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[28],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[30],q[28];
cx q[22],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[21],q[22];
cx q[22],q[21];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[22],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[28],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[34],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[37],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[48],q[50];
cx q[50],q[48];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[43],q[40];
cx q[40],q[43];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[41],q[42];
cx q[42],q[41];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[45],q[44];
cx q[44],q[45];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[54],q[49];
cx q[49],q[54];
cx q[56],q[59];
cx q[59],q[56];
cx q[56],q[59];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[54],q[55];
cx q[49],q[54];
cx q[54],q[49];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[56],q[59];
cx q[59],q[56];
cx q[56],q[59];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[49],q[54];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[57],q[56];
cx q[56],q[57];
cx q[55],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[57],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[68];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[63],q[60];
cx q[60],q[63];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[68],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[68],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[74],q[69];
cx q[69],q[74];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[74],q[75];
cx q[69],q[74];
cx q[74],q[69];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[74],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[92],q[98];
cx q[98],q[92];
cx q[92],q[98];
cx q[77],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[88],q[90];
cx q[90],q[88];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[92],q[98];
cx q[98],q[92];
cx q[92],q[98];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[90],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[88],q[90];
cx q[90],q[88];
cx q[92],q[98];
cx q[98],q[92];
cx q[92],q[98];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[90],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[90],q[88];
cx q[88],q[90];
cx q[90],q[88];
cx q[88],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[88];
cx q[83],q[80];
cx q[80],q[83];
cx q[83],q[80];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[88],q[82];
cx q[82],q[88];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[83],q[80];
cx q[80],q[83];
cx q[83],q[80];
cx q[92],q[98];
cx q[98],q[92];
cx q[92],q[98];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[88],q[90];
cx q[90],q[88];
cx q[88],q[90];
cx q[90],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[93],q[90];
cx q[90],q[93];
cx q[93],q[90];
cx q[88],q[90];
cx q[90],q[88];
cx q[88],q[90];
cx q[90],q[88];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[90];
cx q[90],q[93];
cx q[93],q[90];
cx q[88],q[90];
cx q[90],q[88];
cx q[88],q[90];
cx q[90],q[88];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[90];
cx q[90],q[93];
cx q[88],q[90];
cx q[90],q[93];
cx q[93],q[90];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[96],q[99];
cx q[99],q[96];
cx q[96],q[99];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[94],q[95];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[95],q[94];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[96],q[99];
cx q[99],q[96];
cx q[96],q[99];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[94],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[102],q[87];
cx q[87],q[102];
cx q[102],q[87];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[87];
cx q[87],q[102];
cx q[102],q[87];
cx q[106],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[112],q[118];
cx q[118],q[112];
cx q[112],q[118];
cx q[97],q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[108],q[110];
cx q[110],q[108];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[87];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[112],q[118];
cx q[118],q[112];
cx q[112],q[118];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[108],q[110];
cx q[110],q[108];
cx q[112],q[118];
cx q[118],q[112];
cx q[112],q[118];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[87],q[102];
cx q[102],q[87];
cx q[108],q[102];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[108];
cx q[103],q[100];
cx q[100],q[103];
cx q[103],q[100];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[108],q[110];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[113],q[110];
cx q[110],q[113];
cx q[113],q[110];
cx q[108],q[110];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[101],q[102];
cx q[102],q[101];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
cx q[113],q[110];
cx q[110],q[113];
cx q[108],q[110];
cx q[110],q[113];
cx q[113],q[110];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
cx q[114],q[109];
cx q[115],q[114];
cx q[114],q[115];
cx q[109],q[114];
cx q[106],q[109];
cx q[109],q[106];
cx q[114],q[115];
cx q[115],q[114];
cx q[116],q[119];
cx q[119],q[116];
cx q[116],q[119];
cx q[115],q[116];
cx q[116],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[115],q[114];
cx q[114],q[115];
cx q[115],q[114];
cx q[114],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[106],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
cx q[116],q[119];
cx q[119],q[116];
cx q[116],q[119];
cx q[116],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[114],q[115];
cx q[115],q[114];
cx q[114],q[115];
cx q[115],q[114];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
cx q[117],q[116];
cx q[116],q[117];
cx q[117],q[116];
cx q[116],q[115];
cx q[115],q[116];
cx q[114],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[117],q[116];
cx q[116],q[117];
cx q[117],q[116];
cx q[122],q[107];
cx q[107],q[122];
cx q[122],q[107];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[122],q[107];
cx q[107],q[122];
cx q[122],q[107];
cx q[132],q[138];
cx q[138],q[132];
cx q[132],q[138];
cx q[117],q[132];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
cx q[130],q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[122],q[107];
cx q[107],q[122];
cx q[122],q[107];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[132],q[138];
cx q[138],q[132];
cx q[132],q[138];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
cx q[130],q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[132],q[138];
cx q[138],q[132];
cx q[132],q[138];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[130],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[130],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[128];
cx q[123],q[120];
cx q[120],q[123];
cx q[123],q[120];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[128],q[122];
cx q[122],q[128];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
cx q[130],q[128];
cx q[128],q[130];
cx q[130],q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[123],q[120];
cx q[120],q[123];
cx q[133],q[130];
cx q[130],q[133];
cx q[133],q[130];
cx q[130],q[128];
cx q[128],q[130];
cx q[130],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[121],q[122];
cx q[122],q[121];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[134],q[133];
cx q[133],q[134];
cx q[134],q[133];
cx q[133],q[130];
cx q[130],q[133];
cx q[133],q[130];
cx q[128],q[130];
cx q[130],q[128];
cx q[128],q[130];
cx q[130],q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[123];
cx q[123],q[120];
cx q[124],q[123];
cx q[123],q[124];
cx q[124],q[123];
cx q[125],q[124];
cx q[124],q[125];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[125],q[126];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[134],q[129];
cx q[129],q[134];
cx q[136],q[139];
cx q[139],q[136];
cx q[136],q[139];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[134],q[135];
cx q[129],q[134];
cx q[134],q[129];
cx q[135],q[134];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[125],q[126];
cx q[126],q[125];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[136],q[139];
cx q[139],q[136];
cx q[136],q[139];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[134],q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[129],q[134];
cx q[134],q[129];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[137],q[136];
cx q[136],q[137];
cx q[135],q[136];
cx q[136],q[137];
cx q[137],q[136];
cx q[142],q[127];
cx q[127],q[142];
cx q[142],q[127];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[142],q[127];
cx q[127],q[142];
cx q[142],q[127];
cx q[152],q[158];
cx q[158],q[152];
cx q[152],q[158];
cx q[137],q[152];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[151],q[150];
cx q[150],q[151];
cx q[151],q[150];
cx q[148],q[150];
cx q[150],q[148];
cx q[152],q[158];
cx q[158],q[152];
cx q[152],q[158];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[142],q[127];
cx q[127],q[142];
cx q[142],q[127];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[143],q[140];
cx q[140],q[143];
cx q[143],q[140];
cx q[144],q[143];
cx q[143],q[144];
cx q[144],q[143];
cx q[152],q[158];
cx q[158],q[152];
cx q[152],q[158];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[143],q[140];
cx q[140],q[143];
cx q[143],q[140];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[152],q[158];
cx q[158],q[152];
cx q[152],q[158];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[151];
cx q[151],q[150];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[153],q[150];
cx q[150],q[153];
cx q[153],q[150];
cx q[148],q[150];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[141],q[142];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[154],q[153];
cx q[153],q[154];
cx q[154],q[153];
cx q[153],q[150];
cx q[150],q[153];
cx q[153],q[150];
cx q[148],q[150];
cx q[150],q[148];
cx q[148],q[150];
cx q[150],q[148];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[141],q[142];
cx q[142],q[141];
cx q[142],q[148];
cx q[148],q[142];
cx q[142],q[148];
cx q[155],q[154];
cx q[154],q[155];
cx q[155],q[154];
cx q[154],q[153];
cx q[153],q[154];
cx q[154],q[153];
cx q[153],q[150];
cx q[150],q[153];
cx q[148],q[150];
cx q[150],q[153];
cx q[153],q[150];
cx q[154],q[153];
cx q[153],q[154];
cx q[154],q[153];
cx q[156],q[159];
cx q[159],q[156];
cx q[156],q[159];
cx q[156],q[155];
cx q[155],q[156];
cx q[156],q[155];
cx q[154],q[155];
cx q[155],q[154];
cx q[154],q[155];
cx q[155],q[154];
cx q[154],q[149];
cx q[149],q[146];
cx q[146],q[149];
cx q[149],q[146];
cx q[146],q[149];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[154],q[149];
cx q[149],q[154];
cx q[154],q[149];
cx q[156],q[159];
cx q[159],q[156];
cx q[156],q[159];
cx q[156],q[155];
cx q[155],q[156];
cx q[156],q[155];
cx q[154],q[155];
cx q[155],q[154];
cx q[154],q[155];
cx q[155],q[154];
cx q[154],q[149];
cx q[149],q[154];
cx q[154],q[149];
cx q[149],q[146];
cx q[146],q[149];
cx q[149],q[146];
cx q[146],q[149];
cx q[154],q[149];
cx q[149],q[154];
cx q[154],q[149];
cx q[157],q[156];
cx q[156],q[157];
cx q[157],q[156];
cx q[156],q[155];
cx q[155],q[156];
cx q[154],q[155];
cx q[155],q[156];
cx q[156],q[155];
cx q[162],q[147];
cx q[147],q[162];
cx q[162],q[147];
cx q[165],q[164];
cx q[164],q[165];
cx q[165],q[164];
cx q[166],q[165];
cx q[165],q[166];
cx q[166],q[165];
cx q[165],q[164];
cx q[164],q[165];
cx q[165],q[164];
cx q[162],q[168];
cx q[168],q[162];
cx q[162],q[168];
cx q[162],q[147];
cx q[147],q[162];
cx q[162],q[147];
cx q[172],q[178];
cx q[178],q[172];
cx q[172],q[178];
cx q[172],q[157];
cx q[157],q[172];
cx q[156],q[157];
cx q[157],q[172];
cx q[172],q[157];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[171],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[168],q[170];
cx q[170],q[168];
cx q[162],q[168];
cx q[168],q[162];
cx q[162],q[168];
cx q[162],q[147];
cx q[147],q[162];
cx q[162],q[147];
cx q[171],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[172],q[178];
cx q[178],q[172];
cx q[172],q[178];
cx q[171],q[172];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[171],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[168],q[170];
cx q[170],q[168];
cx q[172],q[178];
cx q[178],q[172];
cx q[172],q[178];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[170],q[171];
cx q[171],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[170],q[168];
cx q[168],q[170];
cx q[170],q[168];
cx q[168],q[162];
cx q[161],q[162];
cx q[162],q[161];
cx q[161],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[162],q[168];
cx q[163],q[160];
cx q[160],q[163];
cx q[163],q[160];
cx q[168],q[162];
cx q[162],q[168];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[171],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[168],q[170];
cx q[170],q[168];
cx q[168],q[170];
cx q[170],q[168];
cx q[162],q[168];
cx q[168],q[162];
cx q[162],q[168];
cx q[161],q[162];
cx q[162],q[161];
cx q[161],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[173],q[170];
cx q[170],q[173];
cx q[173],q[170];
cx q[170],q[168];
cx q[168],q[170];
cx q[170],q[168];
cx q[168],q[162];
cx q[162],q[168];
cx q[161],q[162];
cx q[162],q[161];
cx q[162],q[168];
cx q[168],q[162];
cx q[162],q[168];
cx q[174],q[173];
cx q[173],q[174];
cx q[174],q[173];
cx q[173],q[170];
cx q[170],q[173];
cx q[168],q[170];
cx q[170],q[173];
cx q[173],q[170];
cx q[174],q[173];
cx q[173],q[174];
cx q[174],q[173];
cx q[174],q[169];
cx q[169],q[174];
cx q[174],q[169];
cx q[169],q[166];
cx q[166],q[169];
cx q[169],q[166];
cx q[166],q[169];
cx q[166],q[165];
cx q[165],q[166];
cx q[166],q[165];
cx q[165],q[164];
cx q[164],q[165];
cx q[165],q[164];
cx q[175],q[174];
cx q[174],q[175];
cx q[169],q[174];
cx q[174],q[175];
cx q[176],q[179];
cx q[179],q[176];
cx q[176],q[179];
cx q[175],q[176];
cx q[174],q[175];
cx q[175],q[174];
cx q[176],q[175];
cx q[175],q[174];
cx q[174],q[175];
cx q[175],q[174];
cx q[174],q[169];
cx q[169],q[174];
cx q[174],q[169];
cx q[169],q[166];
cx q[166],q[169];
cx q[169],q[166];
cx q[166],q[169];
cx q[174],q[169];
cx q[169],q[174];
cx q[174],q[169];
cx q[176],q[179];
cx q[179],q[176];
cx q[176],q[179];
cx q[176],q[175];
cx q[175],q[176];
cx q[176],q[175];
cx q[174],q[175];
cx q[175],q[174];
cx q[174],q[175];
cx q[175],q[174];
cx q[174],q[169];
cx q[169],q[174];
cx q[174],q[169];
cx q[166],q[169];
cx q[169],q[166];
cx q[166],q[169];
cx q[181],q[180];
cx q[180],q[181];
cx q[181],q[180];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[182],q[167];
cx q[167],q[182];
cx q[182],q[167];
cx q[166],q[167];
cx q[167],q[166];
cx q[166],q[167];
cx q[167],q[166];
cx q[165],q[166];
cx q[166],q[165];
cx q[167],q[166];
cx q[166],q[167];
cx q[167],q[166];
cx q[182],q[167];
cx q[167],q[182];
cx q[182],q[167];
cx q[183],q[180];
cx q[180],q[183];
cx q[183],q[180];
cx q[181],q[180];
cx q[180],q[181];
cx q[181],q[180];
cx q[181],q[182];
cx q[182],q[181];
cx q[183],q[180];
cx q[180],q[183];
cx q[183],q[180];
cx q[181],q[180];
cx q[180],q[181];
cx q[181],q[180];
cx q[184],q[183];
cx q[183],q[184];
cx q[184],q[183];
cx q[183],q[180];
cx q[180],q[183];
cx q[183],q[180];
cx q[185],q[184];
cx q[184],q[185];
cx q[185],q[184];
cx q[186],q[185];
cx q[185],q[186];
cx q[186],q[185];
cx q[185],q[184];
cx q[184],q[185];
cx q[185],q[184];
cx q[184],q[183];
cx q[183],q[184];
cx q[184],q[183];
cx q[182],q[188];
cx q[188],q[182];
cx q[182],q[188];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[186],q[189];
cx q[189],q[186];
cx q[190],q[188];
cx q[188],q[190];
cx q[190],q[188];
cx q[192],q[177];
cx q[177],q[192];
cx q[192],q[177];
cx q[192],q[191];
cx q[191],q[192];
cx q[190],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[198],q[192];
cx q[192],q[198];
cx q[192],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[190],q[188];
cx q[188],q[190];
cx q[190],q[188];
cx q[188],q[182];
cx q[167],q[182];
cx q[182],q[167];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[180],q[181];
cx q[181],q[180];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[182],q[188];
cx q[188],q[182];
cx q[182],q[188];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[192],q[177];
cx q[177],q[192];
cx q[192],q[177];
cx q[192],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[188],q[190];
cx q[192],q[177];
cx q[177],q[192];
cx q[192],q[177];
cx q[192],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[193],q[190];
cx q[190],q[193];
cx q[193],q[190];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[192],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[194],q[193];
cx q[193],q[194];
cx q[194],q[193];
cx q[193],q[190];
cx q[190],q[193];
cx q[193],q[190];
cx q[194],q[189];
cx q[189],q[186];
cx q[186],q[189];
cx q[185],q[186];
cx q[186],q[185];
cx q[186],q[189];
cx q[189],q[186];
cx q[186],q[189];
cx q[194],q[189];
cx q[189],q[194];
cx q[194],q[189];
cx q[186],q[189];
cx q[189],q[186];
cx q[186],q[189];
cx q[194],q[193];
cx q[193],q[190];
cx q[190],q[193];
cx q[193],q[190];
cx q[190],q[188];
cx q[188],q[190];
cx q[190],q[188];
cx q[188],q[182];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[181],q[180];
cx q[180],q[181];
cx q[181],q[180];
cx q[180],q[183];
cx q[183],q[180];
cx q[180],q[183];
cx q[183],q[180];
cx q[181],q[180];
cx q[180],q[181];
cx q[181],q[180];
cx q[182],q[181];
cx q[181],q[182];
cx q[182],q[181];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[190],q[188];
cx q[188],q[190];
cx q[182],q[188];
cx q[188],q[190];
cx q[190],q[188];
cx q[192],q[191];
cx q[191],q[192];
cx q[192],q[191];
cx q[193],q[190];
cx q[190],q[193];
cx q[193],q[190];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[194],q[189];
cx q[189],q[194];
cx q[194],q[189];
cx q[193],q[194];
cx q[194],q[193];
cx q[193],q[194];
cx q[194],q[193];
cx q[193],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[190],q[191];
cx q[191],q[190];
cx q[193],q[190];
cx q[190],q[193];
cx q[193],q[190];
cx q[195],q[194];
cx q[194],q[195];
cx q[195],q[194];
cx q[193],q[194];
cx q[196],q[195];
cx q[195],q[196];
cx q[196],q[195];
cx q[194],q[195];
