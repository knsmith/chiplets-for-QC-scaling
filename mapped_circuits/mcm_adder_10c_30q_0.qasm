OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
cx q[1],q[2];
cx q[0],q[3];
cx q[3],q[0];
cx q[4],q[5];
cx q[5],q[4];
cx q[2],q[8];
cx q[8],q[2];
cx q[2],q[8];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[8],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[2];
cx q[8],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[8];
cx q[2],q[8];
cx q[8],q[2];
cx q[2],q[8];
cx q[1],q[2];
rz(pi/4) q[1];
rz(-pi/4) q[2];
cx q[1],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(3*pi/4) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[2],q[8];
cx q[8],q[2];
cx q[2],q[8];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(-pi/4) q[0];
x q[8];
cx q[2],q[8];
cx q[8],q[2];
cx q[2],q[8];
cx q[1],q[2];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[8];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[8],q[2];
cx q[2],q[8];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[0],q[1];
rz(pi/4) q[0];
rz(-pi/4) q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-pi/4) q[0];
rz(pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
x q[1];
cx q[2],q[1];
cx q[3],q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[0],q[3];
rz(pi/4) q[3];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[0],q[3];
rz(pi/4) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[0],q[3];
rz(pi/4) q[0];
rz(-pi/4) q[3];
x q[3];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[8];
cx q[8],q[2];
cx q[6],q[9];
cx q[6],q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[5],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[5],q[6];
rz(pi/4) q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[5],q[6];
rz(pi/4) q[5];
rz(-pi/4) q[6];
cx q[5],q[6];
rz(3*pi/4) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
x q[9];
cx q[11],q[12];
cx q[10],q[13];
cx q[13],q[10];
cx q[14],q[15];
cx q[15],q[14];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[6];
cx q[6],q[9];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[12],q[7];
rz(-pi/4) q[7];
cx q[9],q[6];
cx q[6],q[9];
cx q[5],q[6];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[12],q[7];
cx q[7],q[12];
cx q[12],q[7];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[7],q[6];
rz(-pi/4) q[6];
rz(pi/4) q[7];
cx q[9],q[6];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[12],q[7];
cx q[7],q[12];
cx q[12],q[7];
cx q[11],q[12];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[18],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
rz(pi/4) q[12];
cx q[18],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
rz(3*pi/4) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[11],q[10];
rz(pi/4) q[10];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[10],q[11];
cx q[11],q[10];
rz(3*pi/4) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[13];
cx q[10],q[13];
rz(pi/4) q[10];
rz(-pi/4) q[13];
cx q[10],q[13];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[13],q[10];
x q[10];
cx q[11],q[10];
cx q[14],q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[15],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[15];
cx q[14],q[15];
rz(pi/4) q[14];
rz(-pi/4) q[15];
cx q[14],q[15];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/4) q[18];
cx q[12],q[18];
rz(pi/4) q[12];
rz(-pi/4) q[18];
x q[18];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[9],q[6];
rz(-pi/4) q[6];
x q[6];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(pi/4) q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[16],q[19];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[14];
x q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
rz(pi/4) q[15];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[15];
rz(-pi/4) q[15];
cx q[14],q[15];
rz(pi/4) q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
rz(pi/2) q[14];
rz(-pi/4) q[15];
x q[15];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[21],q[22];
cx q[22],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[22],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[22],q[17];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[17];
cx q[16],q[17];
rz(-pi/4) q[17];
cx q[22],q[17];
rz(pi/4) q[17];
cx q[16],q[17];
rz(pi/4) q[16];
rz(-pi/4) q[17];
cx q[17],q[22];
cx q[22],q[17];
cx q[17],q[16];
rz(-pi/4) q[16];
x q[16];
rz(pi/4) q[17];
cx q[17],q[16];
cx q[16],q[17];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[21],q[22];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[28],q[22];
rz(-pi/4) q[22];
cx q[21],q[22];
rz(pi/4) q[22];
cx q[28],q[22];
rz(-pi/4) q[22];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(3*pi/4) q[21];
sx q[21];
rz(pi) q[21];
cx q[21],q[20];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[28];
cx q[22],q[28];
rz(pi/4) q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(-pi/4) q[28];
x q[28];
cx q[28],q[22];
rz(-pi/4) q[22];
cx q[21],q[22];
rz(pi/4) q[22];
cx q[28],q[22];
rz(-pi/4) q[22];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[28];
cx q[22],q[28];
rz(pi/4) q[22];
rz(-pi/4) q[28];
cx q[22],q[28];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
x q[28];
cx q[22],q[28];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[17],q[22];
cx q[16],q[17];
rz(-pi/4) q[22];
cx q[17],q[22];
cx q[16],q[17];
rz(pi/4) q[22];
cx q[17],q[22];
rz(pi/4) q[17];
cx q[16],q[17];
rz(-pi/4) q[16];
rz(-pi/4) q[22];
cx q[17],q[22];
rz(pi/4) q[17];
cx q[17],q[16];
x q[16];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
rz(pi/4) q[15];
cx q[16],q[15];
rz(-pi/4) q[15];
cx q[14],q[15];
rz(pi/4) q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[14],q[15];
rz(-pi/4) q[15];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
x q[14];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[15];
rz(-pi/4) q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[15];
cx q[14],q[15];
rz(pi/4) q[14];
rz(-pi/4) q[15];
cx q[14],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[10],q[13];
rz(-pi/4) q[13];
cx q[13],q[10];
cx q[10],q[13];
cx q[13],q[10];
cx q[11],q[10];
rz(pi/4) q[10];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[11],q[10];
rz(pi/4) q[13];
cx q[13],q[10];
cx q[10],q[13];
cx q[13],q[10];
cx q[11],q[10];
rz(-pi/4) q[10];
rz(pi/4) q[11];
cx q[10],q[11];
cx q[11],q[10];
x q[11];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[10];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[12],q[11];
rz(-pi/4) q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
x q[15];
cx q[14],q[15];
cx q[18],q[12];
rz(pi/4) q[12];
cx q[11],q[12];
rz(pi/4) q[11];
rz(-pi/4) q[12];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[12],q[11];
rz(-pi/4) q[11];
rz(pi/4) q[12];
cx q[12],q[11];
x q[11];
rz(3*pi/4) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[12],q[11];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[12],q[18];
cx q[18],q[12];
cx q[12],q[18];
cx q[22],q[17];
cx q[7],q[12];
rz(-pi/4) q[12];
cx q[12],q[7];
cx q[7],q[12];
cx q[12],q[7];
cx q[6],q[7];
rz(pi/4) q[7];
cx q[12],q[7];
rz(pi/4) q[12];
rz(-pi/4) q[7];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[12];
rz(-pi/4) q[12];
rz(pi/4) q[7];
cx q[7],q[12];
x q[12];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[12];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[5],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
rz(3*pi/4) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[9];
cx q[6],q[9];
rz(pi/4) q[6];
rz(-pi/4) q[9];
cx q[6],q[9];
cx q[5],q[6];
x q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[5],q[6];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[5],q[6];
rz(-pi/4) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[3],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[5];
cx q[4],q[5];
rz(pi/4) q[4];
rz(-pi/4) q[5];
cx q[4],q[5];
cx q[3],q[4];
x q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[3],q[4];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[3],q[4];
rz(-pi/4) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[0],q[3];
rz(pi/4) q[3];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[0],q[3];
rz(pi/4) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[0],q[3];
rz(pi/4) q[0];
rz(-pi/4) q[3];
cx q[0],q[3];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
x q[0];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[4],q[3];
cx q[8],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[8],q[2];
cx q[2],q[1];
rz(-pi/4) q[1];
rz(pi/4) q[2];
cx q[8],q[2];
cx q[2],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[2];
rz(-pi/4) q[8];
cx q[2],q[8];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
x q[8];
cx q[2],q[8];
