OPENQASM 2.0;
include "qelib1.inc";
qreg q[140];
rz(pi/2) q[20];
sx q[20];
rz(-3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[23];
sx q[23];
rz(-3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[26];
sx q[26];
rz(-3*pi/4) q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[27];
sx q[27];
rz(-3*pi/4) q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[29];
sx q[29];
rz(-3*pi/4) q[29];
rz(pi/2) q[30];
sx q[30];
rz(-3*pi/4) q[30];
sx q[30];
rz(pi/2) q[30];
rz(pi/2) q[33];
sx q[33];
rz(-3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[29],q[26];
x q[26];
rz(-pi/2) q[29];
sx q[34];
rz(pi/2) q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
rz(pi/2) q[36];
sx q[36];
rz(-3*pi/4) q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
sx q[35];
rz(pi/2) q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
rz(pi/2) q[39];
sx q[39];
rz(-3*pi/4) q[39];
sx q[39];
rz(pi/2) q[39];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[41];
sx q[41];
rz(-3*pi/4) q[41];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(-3*pi/4) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
rz(pi/2) q[45];
sx q[45];
rz(-3*pi/4) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(pi/2) q[48];
sx q[48];
rz(-3*pi/4) q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
sx q[27];
rz(pi/2) q[27];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
sx q[41];
rz(pi/2) q[41];
sx q[42];
rz(pi/2) q[42];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
sx q[26];
rz(-3*pi/4) q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
sx q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(pi/2) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
sx q[40];
rz(pi/2) q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(pi/2) q[49];
sx q[49];
rz(-3*pi/4) q[49];
sx q[49];
rz(pi/2) q[49];
rz(pi/2) q[50];
sx q[50];
rz(-3*pi/4) q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(pi/2) q[52];
sx q[52];
rz(pi/2) q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
sx q[36];
rz(pi/2) q[36];
rz(pi/2) q[53];
sx q[53];
rz(-3*pi/4) q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(pi/2) q[54];
sx q[54];
rz(-3*pi/4) q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
x q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[27],q[42];
sx q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
rz(-pi) q[42];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
sx q[27];
sx q[50];
sx q[51];
rz(pi/2) q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
rz(pi/2) q[56];
sx q[56];
rz(-3*pi/4) q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[54],q[55];
rz(-pi/2) q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
rz(-pi) q[55];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[37],q[52];
rz(-pi/2) q[37];
x q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[29];
x q[29];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
rz(3*pi/4) q[29];
rz(-pi/2) q[34];
sx q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
rz(-3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
rz(pi/2) q[26];
rz(pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
sx q[34];
rz(pi/2) q[34];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
sx q[36];
rz(-3*pi/4) q[36];
sx q[36];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
rz(pi/2) q[39];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
rz(3*pi/4) q[37];
sx q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
rz(pi/2) q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
sx q[52];
rz(-3*pi/4) q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
sx q[58];
rz(pi/2) q[58];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
rz(pi/2) q[59];
sx q[59];
rz(-3*pi/4) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[62];
sx q[62];
rz(-3*pi/4) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
sx q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
rz(3*pi/4) q[40];
sx q[40];
rz(pi/2) q[40];
rz(-pi) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
sx q[43];
rz(pi/2) q[43];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
rz(-pi/2) q[45];
rz(-pi) q[46];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
sx q[50];
rz(pi/2) q[50];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[64];
sx q[64];
rz(-3*pi/4) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[65];
sx q[65];
rz(-3*pi/4) q[65];
sx q[65];
rz(pi/2) q[65];
rz(pi/2) q[66];
sx q[66];
rz(-3*pi/4) q[66];
sx q[66];
rz(pi/2) q[66];
rz(pi/2) q[67];
sx q[67];
rz(-3*pi/4) q[67];
sx q[67];
rz(pi/2) q[67];
rz(pi/2) q[69];
sx q[69];
rz(-3*pi/4) q[69];
sx q[69];
rz(pi/2) q[69];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
sx q[72];
rz(-3*pi/4) q[72];
rz(pi/2) q[73];
sx q[73];
rz(-3*pi/4) q[73];
rz(pi/2) q[74];
sx q[74];
rz(pi/2) q[74];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[47],q[62];
sx q[47];
cx q[60],q[61];
rz(-pi/2) q[60];
rz(-pi) q[61];
rz(-pi) q[62];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[75];
sx q[75];
rz(-3*pi/4) q[75];
rz(pi/2) q[76];
sx q[76];
rz(-3*pi/4) q[76];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
sx q[78];
rz(pi/2) q[78];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[80];
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
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
sx q[80];
rz(pi/2) q[80];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/2) q[85];
sx q[85];
rz(-3*pi/4) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[87];
sx q[87];
rz(-3*pi/4) q[87];
sx q[87];
rz(pi/2) q[87];
rz(pi/2) q[89];
sx q[89];
rz(pi/2) q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
sx q[85];
rz(3*pi/4) q[85];
sx q[85];
cx q[90],q[88];
cx q[88],q[90];
cx q[90],q[88];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[81],q[82];
sx q[81];
rz(3*pi/4) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
x q[82];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
sx q[63];
rz(pi/2) q[63];
sx q[64];
rz(-3*pi/4) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[88],q[90];
rz(-pi/2) q[88];
sx q[88];
rz(-3*pi/4) q[88];
sx q[88];
rz(pi/2) q[88];
x q[90];
rz(pi/2) q[91];
sx q[91];
rz(-3*pi/4) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[77];
cx q[77],q[92];
cx q[92],q[77];
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
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
cx q[57],q[56];
rz(-pi) q[56];
sx q[57];
cx q[71],q[72];
rz(-pi/2) q[71];
x q[72];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
x q[53];
sx q[54];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
sx q[55];
rz(pi/2) q[55];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[68],q[70];
sx q[68];
rz(-pi) q[70];
cx q[74],q[73];
x q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
sx q[44];
rz(3*pi/4) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
sx q[43];
rz(-pi) q[45];
sx q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/2) q[44];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(3*pi/4) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[49];
rz(-pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
sx q[45];
rz(-3*pi/4) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
sx q[47];
rz(-3*pi/4) q[47];
sx q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi) q[49];
sx q[49];
cx q[61],q[62];
rz(-pi/2) q[61];
rz(-pi) q[62];
sx q[62];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
sx q[60];
rz(-3*pi/4) q[60];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
sx q[68];
rz(pi/2) q[68];
rz(-pi/2) q[74];
sx q[75];
rz(pi/2) q[75];
sx q[76];
rz(pi/2) q[76];
cx q[92],q[77];
rz(-pi) q[77];
rz(-pi/2) q[92];
rz(pi/2) q[94];
sx q[94];
rz(-3*pi/4) q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
sx q[93];
rz(pi/2) q[93];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[86],q[89];
sx q[86];
rz(3*pi/4) q[86];
sx q[86];
rz(pi/2) q[86];
x q[89];
rz(pi/2) q[95];
sx q[95];
rz(-3*pi/4) q[95];
rz(pi/2) q[96];
sx q[96];
rz(pi/2) q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
rz(-pi) q[94];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
sx q[89];
rz(pi/2) q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
x q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
sx q[83];
rz(-3*pi/4) q[83];
rz(pi/2) q[86];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[90];
cx q[90],q[93];
cx q[93],q[90];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
sx q[91];
rz(-3*pi/4) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[92],q[77];
cx q[77],q[92];
cx q[92],q[77];
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
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
sx q[70];
rz(-3*pi/4) q[70];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
rz(3*pi/4) q[71];
sx q[71];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
cx q[57],q[56];
x q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
rz(pi/2) q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
sx q[57];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
rz(-pi) q[42];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
sx q[26];
rz(pi/2) q[26];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[34],q[33];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[52],q[58];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(pi/2) q[51];
sx q[51];
rz(pi/2) q[51];
rz(3*pi/4) q[56];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
rz(-pi) q[48];
sx q[48];
rz(pi/2) q[48];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
sx q[50];
rz(3*pi/4) q[50];
sx q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
rz(-pi/2) q[55];
sx q[55];
rz(-3*pi/4) q[55];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
sx q[56];
rz(pi/2) q[56];
cx q[56],q[59];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[59],q[56];
cx q[56],q[59];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
rz(3*pi/4) q[57];
sx q[57];
rz(pi/2) q[57];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
sx q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
cx q[72],q[78];
rz(3*pi/4) q[73];
sx q[73];
rz(pi/2) q[73];
sx q[74];
rz(pi/2) q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[70],q[73];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(-pi/2) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[49],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
rz(-pi) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
rz(-pi/2) q[54];
sx q[54];
rz(-3*pi/4) q[54];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[73],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
x q[66];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[68],q[70];
sx q[75];
rz(-3*pi/4) q[75];
sx q[75];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[69],q[74];
rz(-pi/2) q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
sx q[64];
rz(pi/2) q[64];
sx q[67];
rz(-3*pi/4) q[67];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[77];
cx q[77],q[92];
cx q[92],q[77];
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
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[49];
sx q[46];
rz(pi/2) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[41],q[42];
rz(pi/2) q[41];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(pi/2) q[43];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
sx q[44];
rz(pi/2) q[44];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi) q[49];
sx q[49];
rz(pi/2) q[49];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
sx q[49];
rz(pi/2) q[49];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[27],q[26];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
sx q[42];
rz(pi/2) q[42];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
rz(pi/2) q[52];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[51];
rz(pi/2) q[50];
sx q[50];
rz(pi/2) q[50];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[53],q[54];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[62];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[63];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
sx q[61];
rz(-3*pi/4) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[47];
cx q[47],q[62];
cx q[62],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
rz(pi/2) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
sx q[44];
rz(pi/2) q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(3*pi/4) q[47];
sx q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(pi/2) q[46];
rz(pi/2) q[68];
rz(-pi/2) q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
sx q[68];
rz(-3*pi/4) q[68];
cx q[70],q[71];
rz(pi/2) q[70];
x q[73];
rz(-pi) q[74];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
sx q[73];
rz(pi/2) q[73];
rz(-pi) q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
sx q[69];
rz(pi/2) q[69];
rz(-pi/2) q[76];
cx q[76],q[79];
rz(pi/2) q[77];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[74],q[75];
rz(pi/2) q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[72],q[57];
cx q[57],q[72];
cx q[72],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[55];
rz(pi/2) q[56];
sx q[56];
rz(pi/2) q[56];
sx q[57];
rz(pi/2) q[57];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
sx q[79];
rz(-3*pi/4) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[92],q[77];
cx q[77],q[92];
cx q[92],q[77];
cx q[76],q[77];
rz(pi/2) q[76];
sx q[76];
rz(pi/2) q[76];
cx q[93],q[90];
rz(-pi) q[90];
sx q[90];
rz(pi/2) q[90];
cx q[91],q[90];
cx q[90],q[91];
cx q[91],q[90];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
rz(-pi/2) q[93];
sx q[93];
rz(-3*pi/4) q[93];
sx q[93];
rz(pi/2) q[93];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[85],q[86];
rz(-pi/2) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
sx q[84];
rz(-3*pi/4) q[84];
rz(-pi) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[80];
cx q[80],q[83];
cx q[83],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[67];
cx q[67],q[82];
cx q[82],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[66],q[67];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
sx q[69];
rz(pi/2) q[69];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[47];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
sx q[70];
rz(pi/2) q[70];
sx q[74];
rz(pi/2) q[74];
sx q[84];
rz(pi/2) q[84];
sx q[85];
rz(pi/2) q[85];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
sx q[95];
rz(3*pi/4) q[95];
sx q[95];
rz(pi/2) q[95];
sx q[96];
rz(pi/2) q[96];
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
rz(-3*pi/4) q[99];
sx q[99];
rz(pi/2) q[99];
cx q[96],q[99];
cx q[99],q[96];
cx q[96],q[99];
rz(pi/2) q[100];
sx q[100];
rz(-3*pi/4) q[100];
rz(pi/2) q[101];
sx q[101];
rz(-3*pi/4) q[101];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[103],q[100];
cx q[100],q[103];
cx q[103],q[100];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
sx q[100];
rz(pi/2) q[100];
sx q[103];
rz(pi/2) q[103];
cx q[103],q[100];
cx q[100],q[103];
rz(pi/2) q[104];
sx q[104];
rz(-3*pi/4) q[104];
sx q[104];
rz(pi/2) q[104];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
rz(pi/2) q[107];
sx q[107];
rz(-3*pi/4) q[107];
sx q[107];
rz(pi/2) q[107];
rz(pi/2) q[109];
sx q[109];
rz(-3*pi/4) q[109];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[101],q[102];
sx q[101];
x q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
x q[100];
cx q[100],q[103];
rz(-pi/2) q[101];
rz(3*pi/4) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[103],q[100];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
sx q[100];
cx q[103],q[100];
cx q[100],q[103];
cx q[103],q[100];
rz(-3*pi/4) q[103];
sx q[103];
rz(pi/2) q[103];
rz(pi/2) q[111];
sx q[111];
rz(-3*pi/4) q[111];
rz(pi/2) q[112];
sx q[112];
rz(-3*pi/4) q[112];
cx q[112],q[97];
cx q[97],q[112];
cx q[112],q[97];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[108],q[110];
sx q[108];
rz(3*pi/4) q[108];
sx q[108];
rz(pi/2) q[108];
rz(-pi) q[110];
sx q[110];
rz(pi/2) q[110];
sx q[97];
rz(pi/2) q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[99];
cx q[99],q[96];
cx q[96],q[99];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[89],q[94];
rz(pi/2) q[89];
cx q[96],q[99];
cx q[99],q[96];
cx q[96],q[99];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[89];
cx q[89],q[94];
cx q[94],q[89];
sx q[94];
rz(pi/2) q[94];
rz(pi/2) q[113];
sx q[113];
rz(-3*pi/4) q[113];
sx q[113];
rz(pi/2) q[113];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
cx q[106],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[105],q[106];
sx q[105];
rz(3*pi/4) q[105];
sx q[105];
rz(pi/2) q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
x q[105];
rz(-pi) q[106];
sx q[106];
cx q[105],q[106];
cx q[106],q[105];
sx q[105];
rz(-3*pi/4) q[105];
sx q[105];
rz(pi/2) q[105];
rz(pi/2) q[115];
sx q[115];
rz(-3*pi/4) q[115];
rz(pi/2) q[116];
sx q[116];
rz(pi/2) q[116];
cx q[116],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[115],q[114];
cx q[114],q[115];
cx q[115],q[114];
cx q[114],q[109];
x q[109];
sx q[114];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
rz(3*pi/4) q[109];
sx q[109];
rz(pi/2) q[109];
cx q[106],q[109];
cx q[109],q[106];
cx q[106],q[109];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[106];
sx q[115];
rz(pi/2) q[115];
sx q[116];
rz(pi/2) q[116];
cx q[116],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[114],q[115];
sx q[114];
rz(3*pi/4) q[114];
sx q[114];
cx q[114],q[113];
cx q[113],q[114];
cx q[114],q[113];
rz(pi/2) q[113];
x q[115];
cx q[115],q[116];
rz(pi/2) q[115];
rz(pi/2) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[112],q[118];
cx q[118],q[112];
cx q[112],q[118];
cx q[112],q[111];
rz(-pi) q[111];
sx q[111];
rz(pi/2) q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
rz(-pi) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
sx q[101];
rz(3*pi/4) q[101];
sx q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
rz(pi/2) q[100];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
cx q[102],q[87];
sx q[112];
rz(3*pi/4) q[112];
sx q[112];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[112],q[97];
sx q[118];
rz(pi/2) q[118];
cx q[112],q[118];
cx q[118],q[112];
cx q[112],q[118];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[108],q[110];
rz(pi/2) q[108];
rz(pi/2) q[112];
sx q[118];
rz(3*pi/4) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[87],q[102];
cx q[102],q[87];
cx q[102],q[108];
cx q[108],q[102];
cx q[102],q[108];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[102],q[87];
cx q[110],q[108];
cx q[108],q[110];
cx q[110],q[108];
cx q[87],q[102];
cx q[102],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[89];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
x q[97];
cx q[112],q[97];
cx q[97],q[112];
cx q[112],q[97];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[111],q[110];
rz(pi/2) q[111];
sx q[111];
rz(pi/2) q[111];
rz(pi/2) q[119];
sx q[119];
rz(-3*pi/4) q[119];
sx q[119];
rz(pi/2) q[119];
cx q[116],q[119];
cx q[119],q[116];
cx q[116],q[119];
rz(pi/2) q[120];
sx q[120];
rz(-3*pi/4) q[120];
rz(pi/2) q[121];
sx q[121];
rz(-3*pi/4) q[121];
rz(pi/2) q[122];
sx q[122];
rz(pi/2) q[122];
rz(pi/2) q[123];
sx q[123];
rz(pi/2) q[123];
cx q[123],q[120];
cx q[120],q[123];
cx q[123],q[120];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
sx q[120];
rz(pi/2) q[120];
sx q[123];
rz(pi/2) q[123];
rz(pi/2) q[124];
sx q[124];
rz(-3*pi/4) q[124];
sx q[124];
rz(pi/2) q[124];
rz(pi/2) q[125];
sx q[125];
rz(pi/2) q[125];
rz(pi/2) q[129];
sx q[129];
rz(-3*pi/4) q[129];
cx q[126],q[129];
cx q[129],q[126];
cx q[126],q[129];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
sx q[125];
rz(pi/2) q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
x q[125];
rz(-pi/2) q[126];
cx q[130],q[128];
cx q[128],q[130];
cx q[130],q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
cx q[121],q[122];
rz(-pi/2) q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
sx q[120];
rz(-3*pi/4) q[120];
sx q[120];
rz(-pi) q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
sx q[121];
rz(pi/2) q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[123];
rz(-pi/2) q[120];
sx q[120];
rz(-3*pi/4) q[120];
rz(pi/2) q[121];
rz(-pi) q[123];
sx q[123];
rz(pi/2) q[123];
cx q[123],q[120];
cx q[120],q[123];
cx q[123],q[120];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
sx q[123];
rz(pi/2) q[123];
rz(pi/2) q[131];
sx q[131];
rz(-3*pi/4) q[131];
rz(pi/2) q[132];
sx q[132];
rz(-3*pi/4) q[132];
cx q[132],q[117];
cx q[117],q[132];
cx q[132],q[117];
sx q[117];
rz(pi/2) q[117];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
sx q[128];
rz(-pi) q[130];
rz(pi/2) q[133];
sx q[133];
rz(-3*pi/4) q[133];
sx q[133];
rz(pi/2) q[133];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
x q[129];
cx q[129],q[126];
cx q[126],q[129];
rz(-pi/2) q[126];
cx q[125],q[126];
cx q[126],q[125];
sx q[125];
rz(-3*pi/4) q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
sx q[124];
rz(pi/2) q[124];
sx q[129];
rz(-3*pi/4) q[129];
sx q[129];
rz(pi/2) q[135];
sx q[135];
rz(-3*pi/4) q[135];
rz(pi/2) q[136];
sx q[136];
rz(pi/2) q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[135],q[134];
rz(-pi) q[134];
sx q[134];
rz(pi/2) q[134];
cx q[134],q[133];
cx q[133],q[134];
cx q[134],q[133];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
cx q[126],q[129];
rz(pi/2) q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
sx q[125];
rz(pi/2) q[125];
cx q[134],q[133];
cx q[133],q[134];
cx q[134],q[133];
rz(pi/2) q[133];
sx q[135];
rz(3*pi/4) q[135];
sx q[135];
sx q[136];
rz(pi/2) q[136];
cx q[136],q[135];
cx q[135],q[136];
cx q[136],q[135];
cx q[134],q[135];
sx q[134];
x q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[134],q[129];
cx q[129],q[134];
cx q[134],q[129];
cx q[129],q[126];
rz(pi/2) q[129];
sx q[129];
rz(pi/2) q[129];
rz(3*pi/4) q[135];
sx q[135];
rz(pi/2) q[135];
rz(pi/2) q[136];
rz(pi/2) q[138];
sx q[138];
rz(pi/2) q[138];
cx q[132],q[138];
cx q[138],q[132];
cx q[132],q[138];
cx q[132],q[131];
x q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[130],q[128];
cx q[128],q[130];
cx q[130],q[128];
cx q[128],q[122];
rz(-pi) q[122];
sx q[122];
rz(pi/2) q[122];
sx q[128];
rz(3*pi/4) q[128];
sx q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
rz(pi/2) q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
rz(3*pi/4) q[130];
sx q[130];
rz(pi/2) q[130];
sx q[131];
rz(pi/2) q[131];
rz(-pi/2) q[132];
sx q[132];
rz(-3*pi/4) q[132];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
sx q[131];
rz(pi/2) q[131];
cx q[132],q[117];
cx q[117],q[132];
cx q[132],q[117];
cx q[117],q[116];
x q[116];
cx q[116],q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[115],q[114];
cx q[114],q[115];
cx q[115],q[114];
cx q[114],q[109];
cx q[109],q[114];
cx q[114],q[109];
cx q[109],q[106];
rz(pi/2) q[109];
sx q[109];
rz(pi/2) q[109];
cx q[114],q[115];
rz(pi/2) q[114];
sx q[114];
rz(pi/2) q[114];
sx q[116];
rz(pi/2) q[116];
sx q[117];
rz(3*pi/4) q[117];
sx q[117];
rz(pi/2) q[117];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
rz(pi/2) q[128];
cx q[122],q[128];
cx q[128],q[122];
cx q[122],q[128];
sx q[122];
rz(pi/2) q[122];
sx q[138];
rz(pi/2) q[138];
cx q[132],q[138];
cx q[138],q[132];
cx q[132],q[138];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[128],q[130];
rz(pi/2) q[128];
sx q[128];
rz(pi/2) q[128];