OPENQASM 2.0;
include "qelib1.inc";
qreg q[180];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[0],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[0],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[10],q[9];
cx q[9],q[10];
cx q[7],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[13];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[35],q[39];
cx q[39],q[35];
cx q[35],q[39];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[12];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[8];
cx q[16],q[11];
cx q[11],q[16];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[8],q[13];
cx q[13],q[8];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[6],q[18];
cx q[18],q[6];
cx q[6],q[18];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[16],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[13];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[8];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[8],q[13];
cx q[13],q[8];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[69],q[67];
cx q[67],q[69];
cx q[69],q[67];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[12],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[62],q[16];
cx q[16],q[62];
cx q[62],q[16];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[70],q[69];
cx q[69],q[70];
cx q[70],q[69];
cx q[69],q[67];
cx q[67],q[69];
cx q[69],q[67];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
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
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[17];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[68],q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[107],q[100];
cx q[100],q[107];
cx q[107],q[100];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[109],q[107];
cx q[107],q[109];
cx q[109],q[107];
cx q[107],q[100];
cx q[100],q[107];
cx q[107],q[100];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[76];
cx q[76],q[122];
cx q[122],q[76];
cx q[71],q[76];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[122],q[76];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[19],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[29],q[27];
cx q[27],q[29];
cx q[20],q[27];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[66],q[17];
cx q[17],q[66];
cx q[15],q[17];
cx q[15],q[19];
cx q[17],q[66];
cx q[19],q[15];
cx q[15],q[19];
cx q[66],q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[68],q[64];
cx q[64],q[68];
cx q[65],q[64];
cx q[64],q[68];
cx q[66],q[65];
cx q[65],q[66];
cx q[68],q[64];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[76],q[122];
cx q[71],q[76];
cx q[76],q[122];
cx q[122],q[76];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[127];
cx q[127],q[120];
cx q[120],q[127];
cx q[127],q[120];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[76];
cx q[76],q[122];
cx q[122],q[76];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[17],q[66];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[68],q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[76],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[129],q[127];
cx q[127],q[129];
cx q[129],q[127];
cx q[127],q[120];
cx q[120],q[127];
cx q[127],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[122],q[76];
cx q[76],q[122];
cx q[122],q[76];
cx q[123],q[122];
cx q[122],q[123];
cx q[123],q[122];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[122],q[76];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[65],q[66];
cx q[66],q[78];
cx q[74],q[73];
cx q[73],q[74];
cx q[74],q[73];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[76],q[122];
cx q[122],q[76];
cx q[76],q[71];
cx q[71],q[76];
cx q[78],q[66];
cx q[66],q[78];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[36],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[19];
cx q[19],q[29];
cx q[29],q[19];
cx q[19],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[66],q[17];
cx q[17],q[66];
cx q[66],q[17];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[122],q[76];
cx q[76],q[122];
cx q[122],q[76];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[130],q[129];
cx q[129],q[130];
cx q[130],q[129];
cx q[129],q[127];
cx q[127],q[129];
cx q[129],q[127];
cx q[127],q[120];
cx q[120],q[127];
cx q[121],q[120];
cx q[120],q[127];
cx q[127],q[120];
cx q[129],q[127];
cx q[127],q[129];
cx q[129],q[127];
cx q[136],q[131];
cx q[131],q[136];
cx q[136],q[131];
cx q[131],q[130];
cx q[130],q[131];
cx q[129],q[130];
cx q[130],q[131];
cx q[131],q[130];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[133],q[132];
cx q[132],q[133];
cx q[133],q[132];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[126],q[77];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[136],q[131];
cx q[131],q[136];
cx q[136],q[131];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[136],q[131];
cx q[131],q[136];
cx q[136],q[131];
cx q[77],q[126];
cx q[126],q[77];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[75],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[89],q[79];
cx q[79],q[89];
cx q[89],q[79];
cx q[75],q[79];
cx q[79],q[75];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[36],q[82];
cx q[82],q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[89],q[87];
cx q[87],q[89];
cx q[89],q[87];
cx q[89],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[126],q[77];
cx q[77],q[126];
cx q[126],q[77];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[132],q[133];
cx q[133],q[132];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[126],q[77];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[133],q[128];
cx q[128],q[133];
cx q[136],q[131];
cx q[131],q[136];
cx q[136],q[131];
cx q[77],q[126];
cx q[126],q[77];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[75],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[126],q[77];
cx q[77],q[126];
cx q[89],q[79];
cx q[79],q[89];
cx q[89],q[79];
cx q[75],q[79];
cx q[79],q[75];
cx q[89],q[87];
cx q[87],q[89];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[36],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[35],q[39];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[39],q[35];
cx q[35],q[39];
cx q[37],q[35];
cx q[35],q[37];
cx q[37],q[35];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[36],q[82];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[37],q[35];
cx q[35],q[37];
cx q[37],q[35];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[89],q[87];
cx q[87],q[89];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[89],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[75],q[77];
cx q[77],q[126];
cx q[126],q[77];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[132],q[133];
cx q[133],q[132];
cx q[132],q[131];
cx q[131],q[132];
cx q[132],q[131];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[126],q[77];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[133],q[132];
cx q[132],q[133];
cx q[133],q[132];
cx q[77],q[126];
cx q[126],q[77];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[126],q[77];
cx q[75],q[79];
cx q[77],q[126];
cx q[126],q[77];
cx q[79],q[75];
cx q[75],q[79];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[89],q[79];
cx q[79],q[89];
cx q[89],q[79];
cx q[75],q[79];
cx q[79],q[75];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[89],q[87];
cx q[87],q[89];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[89],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[126],q[77];
cx q[77],q[126];
cx q[126],q[77];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[126],q[77];
cx q[134],q[133];
cx q[133],q[134];
cx q[134],q[133];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[128],q[124];
cx q[124],q[128];
cx q[77],q[126];
cx q[126],q[77];
cx q[77],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[75],q[79];
cx q[79],q[75];
cx q[75],q[79];
cx q[89],q[79];
cx q[79],q[89];
cx q[89],q[79];
cx q[89],q[87];
cx q[87],q[80];
cx q[80],q[87];
cx q[87],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[36];
cx q[36],q[82];
cx q[82],q[36];
cx q[31],q[36];
cx q[36],q[31];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[36],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[90],q[89];
cx q[89],q[90];
cx q[90],q[89];
cx q[89],q[87];
cx q[87],q[89];
cx q[80],q[87];
cx q[87],q[89];
cx q[89],q[87];
cx q[89],q[79];
cx q[79],q[89];
cx q[89],q[79];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[37],q[35];
cx q[35],q[37];
cx q[37],q[35];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[137],q[135];
cx q[135],q[137];
cx q[137],q[135];
cx q[135],q[134];
cx q[134],q[135];
cx q[126],q[138];
cx q[138],q[126];
cx q[126],q[138];
cx q[126],q[77];
cx q[77],q[126];
cx q[126],q[77];
cx q[77],q[75];
cx q[75],q[77];
cx q[79],q[75];
cx q[75],q[77];
cx q[77],q[75];
cx q[126],q[77];
cx q[77],q[126];
cx q[126],q[77];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[128];
cx q[128],q[124];
cx q[133],q[128];
cx q[128],q[133];
cx q[133],q[128];
cx q[133],q[134];
cx q[134],q[135];
cx q[135],q[134];
cx q[135],q[139];
cx q[139],q[135];
cx q[135],q[139];
cx q[137],q[135];
cx q[135],q[137];
cx q[137],q[135];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[135],q[139];
cx q[139],q[135];
cx q[135],q[139];
cx q[137],q[135];
cx q[135],q[137];
cx q[137],q[135];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[142],q[96];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[96],q[142];
cx q[142],q[96];
cx q[96],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[88],q[93];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[93],q[88];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[93],q[92];
cx q[92],q[93];
cx q[96],q[91];
cx q[91],q[96];
cx q[96],q[91];
cx q[142],q[96];
cx q[96],q[142];
cx q[142],q[96];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[147],q[149];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[142],q[96];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[96],q[142];
cx q[142],q[96];
cx q[96],q[91];
cx q[91],q[96];
cx q[96],q[91];
cx q[142],q[96];
cx q[91],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[88],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[37],q[35];
cx q[35],q[37];
cx q[37],q[35];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[86],q[37];
cx q[37],q[86];
cx q[86],q[37];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[93],q[88];
cx q[88],q[93];
cx q[84],q[88];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[86],q[98];
cx q[88],q[93];
cx q[93],q[88];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[96],q[142];
cx q[142],q[96];
cx q[96],q[91];
cx q[91],q[96];
cx q[92],q[91];
cx q[91],q[96];
cx q[93],q[92];
cx q[92],q[93];
cx q[96],q[91];
cx q[142],q[96];
cx q[96],q[142];
cx q[142],q[96];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[147],q[149];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[139],q[135];
cx q[135],q[139];
cx q[139],q[135];
cx q[135],q[139];
cx q[137],q[135];
cx q[135],q[137];
cx q[137],q[135];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[135],q[139];
cx q[139],q[135];
cx q[135],q[139];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[142],q[96];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[149],q[139];
cx q[139],q[149];
cx q[149],q[139];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[96],q[142];
cx q[142],q[96];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[96],q[91];
cx q[91],q[96];
cx q[96],q[91];
cx q[142],q[96];
cx q[91],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[96],q[142];
cx q[91],q[96];
cx q[96],q[142];
cx q[142],q[96];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[141],q[140];
cx q[140],q[141];
cx q[141],q[140];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[142],q[96];
cx q[96],q[142];
cx q[142],q[96];
cx q[143],q[142];
cx q[142],q[143];
cx q[143],q[142];
cx q[96],q[91];
cx q[91],q[96];
cx q[96],q[91];
cx q[142],q[96];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[96],q[142];
cx q[98],q[86];
cx q[86],q[98];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[88],q[84];
cx q[84],q[88];
cx q[88],q[84];
cx q[93],q[88];
cx q[88],q[93];
cx q[93],q[88];
cx q[92],q[93];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[99],q[109];
cx q[109],q[99];
cx q[99],q[109];
cx q[109],q[99];
cx q[109],q[107];
cx q[107],q[109];
cx q[109],q[107];
cx q[107],q[100];
cx q[100],q[107];
cx q[107],q[100];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[109],q[99];
cx q[95],q[94];
cx q[94],q[93];
cx q[93],q[94];
cx q[94],q[93];
cx q[93],q[92];
cx q[92],q[93];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[92];
cx q[92],q[91];
cx q[91],q[96];
cx q[96],q[142];
cx q[142],q[96];
cx q[142],q[141];
cx q[141],q[142];
cx q[142],q[141];
cx q[99],q[109];
cx q[109],q[99];
cx q[109],q[107];
cx q[107],q[109];
cx q[109],q[107];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[109],q[99];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[99],q[109];
cx q[109],q[99];
cx q[109],q[107];
cx q[107],q[109];
cx q[109],q[107];
cx q[107],q[100];
cx q[100],q[107];
cx q[107],q[100];
cx q[150],q[149];
cx q[149],q[150];
cx q[150],q[149];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[147],q[140];
cx q[140],q[147];
cx q[141],q[140];
cx q[140],q[147];
cx q[147],q[140];
cx q[149],q[147];
cx q[147],q[149];
cx q[149],q[147];
cx q[156],q[151];
cx q[151],q[156];
cx q[156],q[151];
cx q[151],q[150];
cx q[150],q[151];
cx q[149],q[150];
cx q[150],q[151];
cx q[151],q[150];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[153],q[152];
cx q[152],q[153];
cx q[153],q[152];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[146],q[97];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[156],q[151];
cx q[151],q[156];
cx q[156],q[151];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[156],q[151];
cx q[151],q[156];
cx q[156],q[151];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[94],q[95];
cx q[95],q[94];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[146],q[97];
cx q[95],q[99];
cx q[97],q[146];
cx q[146],q[97];
cx q[146],q[145];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[152],q[153];
cx q[153],q[152];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[146],q[97];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[153],q[148];
cx q[148],q[153];
cx q[156],q[151];
cx q[151],q[156];
cx q[156],q[151];
cx q[97],q[146];
cx q[146],q[97];
cx q[99],q[95];
cx q[95],q[99];
cx q[97],q[95];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[99],q[109];
cx q[95],q[99];
cx q[99],q[95];
cx q[109],q[99];
cx q[109],q[107];
cx q[107],q[109];
cx q[109],q[107];
cx q[99],q[95];
cx q[95],q[99];
cx q[109],q[99];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[146],q[97];
cx q[97],q[146];
cx q[146],q[97];
cx q[146],q[145];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[144],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[152],q[153];
cx q[153],q[152];
cx q[152],q[151];
cx q[151],q[152];
cx q[152],q[151];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[146],q[97];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[148],q[144];
cx q[144],q[148];
cx q[97],q[146];
cx q[146],q[97];
cx q[99],q[109];
cx q[109],q[99];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[111],q[110];
cx q[110],q[111];
cx q[111],q[110];
cx q[112],q[111];
cx q[111],q[112];
cx q[112],q[111];
cx q[116],q[111];
cx q[111],q[116];
cx q[116],q[111];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[109],q[99];
cx q[95],q[97];
cx q[97],q[95];
cx q[146],q[97];
cx q[97],q[146];
cx q[146],q[97];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[145],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[152],q[153];
cx q[153],q[152];
cx q[153],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[148],q[144];
cx q[144],q[148];
cx q[148],q[144];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[154],q[153];
cx q[153],q[154];
cx q[154],q[153];
cx q[153],q[148];
cx q[148],q[153];
cx q[99],q[109];
cx q[109],q[99];
cx q[110],q[109];
cx q[109],q[110];
cx q[110],q[109];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[146],q[97];
cx q[97],q[95];
cx q[95],q[97];
cx q[97],q[95];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[99],q[109];
cx q[109],q[99];
cx q[99],q[109];
cx q[109],q[99];
cx q[95],q[99];
cx q[99],q[95];
cx q[95],q[99];
cx q[157],q[155];
cx q[155],q[157];
cx q[157],q[155];
cx q[146],q[158];
cx q[158],q[146];
cx q[146],q[158];
cx q[146],q[97];
cx q[97],q[146];
cx q[95],q[97];
cx q[97],q[146];
cx q[146],q[97];
cx q[146],q[145];
cx q[145],q[146];
cx q[146],q[145];
cx q[145],q[144];
cx q[144],q[145];
cx q[145],q[144];
cx q[144],q[148];
cx q[148],q[153];
cx q[153],q[148];
cx q[154],q[153];
cx q[153],q[154];
cx q[154],q[153];
cx q[154],q[155];
cx q[155],q[159];
cx q[159],q[155];
cx q[155],q[159];
cx q[157],q[155];
cx q[155],q[157];
cx q[157],q[155];
cx q[162],q[116];
cx q[116],q[162];
cx q[162],q[116];
cx q[116],q[111];
cx q[111],q[116];
cx q[116],q[111];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[162],q[116];
cx q[116],q[162];
cx q[162],q[116];
cx q[165],q[164];
cx q[164],q[165];
cx q[165],q[164];
cx q[169],q[159];
cx q[159],q[169];
cx q[169],q[159];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[160],q[167];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[162],q[116];
cx q[116],q[162];
cx q[162],q[116];
cx q[116],q[111];
cx q[111],q[116];
cx q[116],q[111];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[169],q[159];
cx q[159],q[169];
cx q[169],q[159];
cx q[159],q[155];
cx q[155],q[159];
cx q[159],q[155];
cx q[155],q[159];
cx q[157],q[155];
cx q[155],q[157];
cx q[157],q[155];
cx q[169],q[159];
cx q[159],q[169];
cx q[169],q[159];
cx q[155],q[159];
cx q[159],q[155];
cx q[155],q[159];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[160],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[161],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[162],q[116];
cx q[116],q[162];
cx q[162],q[116];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[163],q[162];
cx q[162],q[163];
cx q[163],q[162];
cx q[164],q[163];
cx q[163],q[164];
cx q[164],q[163];
cx q[165],q[164];
cx q[164],q[165];
cx q[165],q[164];
cx q[168],q[164];
cx q[164],q[168];
cx q[168],q[164];
cx q[169],q[167];
cx q[167],q[169];
cx q[160],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[159],q[169];
cx q[169],q[159];
cx q[155],q[159];
cx q[159],q[155];
cx q[155],q[159];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[160],q[161];
cx q[161],q[160];
cx q[162],q[161];
cx q[161],q[162];
cx q[167],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[169],q[159];
cx q[159],q[169];
cx q[169],q[159];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[160],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[162],q[163];
cx q[163],q[162];
cx q[162],q[163];
cx q[163],q[162];
cx q[162],q[161];
cx q[161],q[162];
cx q[162],q[161];
cx q[170],q[169];
cx q[169],q[170];
cx q[170],q[169];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[167],q[160];
cx q[160],q[167];
cx q[161],q[160];
cx q[160],q[167];
cx q[167],q[160];
cx q[169],q[167];
cx q[167],q[169];
cx q[169],q[167];
cx q[176],q[171];
cx q[171],q[176];
cx q[176],q[171];
cx q[171],q[170];
cx q[170],q[171];
cx q[169],q[170];
cx q[170],q[171];
cx q[171],q[170];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[173],q[172];
cx q[172],q[173];
cx q[173],q[172];
cx q[168],q[173];
cx q[173],q[168];
cx q[168],q[164];
cx q[164],q[168];
cx q[168],q[164];
cx q[176],q[171];
cx q[171],q[176];
cx q[176],q[171];
cx q[172],q[171];
cx q[171],q[172];
cx q[172],q[171];
cx q[172],q[173];
cx q[173],q[172];
cx q[168],q[173];
cx q[173],q[168];
cx q[173],q[172];
cx q[172],q[173];
cx q[173],q[172];
cx q[174],q[173];
cx q[173],q[174];
cx q[174],q[173];
cx q[176],q[171];
cx q[171],q[176];
cx q[176],q[171];
cx q[172],q[171];
cx q[171],q[176];
cx q[173],q[172];
cx q[172],q[173];
cx q[176],q[171];
cx q[171],q[176];
cx q[176],q[171];
cx q[171],q[172];
cx q[172],q[173];
cx q[173],q[172];
cx q[177],q[175];
cx q[175],q[177];
cx q[177],q[175];
cx q[175],q[174];
cx q[174],q[175];
cx q[175],q[174];
cx q[173],q[174];
cx q[177],q[175];
cx q[175],q[177];
cx q[177],q[175];
cx q[174],q[175];
cx q[175],q[179];
