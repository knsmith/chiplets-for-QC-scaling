OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg m0[24];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/4) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/4) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
sx q[2];
rz(-pi/2) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(-pi/4) q[3];
sx q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
rz(-pi/4) q[4];
sx q[4];
rz(-pi/2) q[4];
rz(-pi/2) q[5];
sx q[5];
rz(-pi/4) q[5];
sx q[5];
rz(-pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
rz(-pi/4) q[6];
sx q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
rz(-pi/4) q[7];
sx q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[8];
sx q[8];
rz(-pi/4) q[8];
sx q[8];
rz(-pi/2) q[8];
rz(-pi/2) q[10];
sx q[10];
rz(-pi/4) q[10];
sx q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[11];
sx q[11];
rz(-pi/4) q[11];
sx q[11];
rz(-pi/2) q[11];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/4) q[12];
sx q[12];
rz(-pi/2) q[12];
rz(-pi/2) q[13];
sx q[13];
rz(-pi/4) q[13];
sx q[13];
rz(-pi/2) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(-pi/4) q[14];
sx q[14];
rz(-pi/2) q[14];
rz(-pi/2) q[15];
sx q[15];
rz(-pi/4) q[15];
sx q[15];
rz(-pi/2) q[15];
rz(-pi/2) q[16];
sx q[16];
rz(-pi/4) q[16];
sx q[16];
rz(-pi/2) q[16];
rz(-pi/2) q[17];
sx q[17];
rz(-pi/4) q[17];
sx q[17];
rz(-pi/2) q[17];
rz(-pi/2) q[20];
sx q[20];
rz(-pi/4) q[20];
sx q[20];
rz(-pi/2) q[20];
rz(-pi/2) q[21];
sx q[21];
rz(-pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-pi/2) q[22];
sx q[22];
rz(-pi/4) q[22];
sx q[22];
rz(-pi/2) q[22];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/4) q[23];
sx q[23];
rz(-pi/2) q[23];
rz(-pi/2) q[24];
sx q[24];
rz(-pi/4) q[24];
sx q[24];
rz(-pi/2) q[24];
rz(-pi/2) q[25];
sx q[25];
rz(-pi/4) q[25];
sx q[25];
rz(-pi/2) q[25];
rz(-pi/2) q[26];
sx q[26];
rz(-pi/4) q[26];
sx q[26];
rz(-pi/2) q[26];
cx q[26],q[25];
rz(-pi/2) q[25];
cx q[26],q[25];
cx q[25],q[24];
rz(-pi/2) q[24];
cx q[25],q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
cx q[24],q[23];
cx q[23],q[20];
rz(-pi/2) q[20];
cx q[23],q[20];
cx q[20],q[21];
rz(-pi/2) q[21];
cx q[20],q[21];
cx q[21],q[22];
rz(-pi/2) q[22];
cx q[21],q[22];
cx q[22],q[17];
rz(-pi/2) q[17];
cx q[22],q[17];
cx q[17],q[16];
rz(-pi/2) q[16];
cx q[17],q[16];
cx q[16],q[15];
rz(-pi/2) q[15];
cx q[16],q[15];
cx q[15],q[14];
rz(-pi/2) q[14];
cx q[15],q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
cx q[14],q[13];
cx q[13],q[10];
rz(-pi/2) q[10];
cx q[13],q[10];
cx q[10],q[11];
rz(-pi/2) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-pi/2) q[12];
cx q[11],q[12];
cx q[12],q[7];
rz(-pi/2) q[7];
cx q[12],q[7];
cx q[7],q[6];
rz(-pi/2) q[6];
cx q[7],q[6];
cx q[6],q[5];
rz(-pi/2) q[5];
cx q[6],q[5];
cx q[5],q[4];
rz(-pi/2) q[4];
cx q[5],q[4];
cx q[4],q[3];
rz(-pi/2) q[3];
cx q[4],q[3];
cx q[3],q[0];
rz(-pi/2) q[0];
cx q[3],q[0];
cx q[0],q[1];
rz(-pi/2) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-pi/2) q[2];
cx q[1],q[2];
cx q[2],q[8];
rz(-pi/2) q[8];
cx q[2],q[8];
measure q[26] -> m0[0];
measure q[25] -> m0[1];
measure q[24] -> m0[2];
measure q[23] -> m0[3];
measure q[20] -> m0[4];
measure q[21] -> m0[5];
measure q[22] -> m0[6];
measure q[17] -> m0[7];
measure q[16] -> m0[8];
measure q[15] -> m0[9];
measure q[14] -> m0[10];
measure q[13] -> m0[11];
measure q[10] -> m0[12];
measure q[11] -> m0[13];
measure q[12] -> m0[14];
measure q[7] -> m0[15];
measure q[6] -> m0[16];
measure q[5] -> m0[17];
measure q[4] -> m0[18];
measure q[3] -> m0[19];
measure q[0] -> m0[20];
measure q[1] -> m0[21];
measure q[2] -> m0[22];
measure q[8] -> m0[23];
