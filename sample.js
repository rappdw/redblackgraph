require('console.table');

var people = [
  'Jiovan Melendez',
  'Linette Inez Williams',
  'Marianna Rapp',
  'Michael Joseph Rapp',
  'Samuel Stewart',
  'John Davis Williams',
  'Daniel Rapp',
  'Barbara Valcarcel',
  'Emeline Chidester',
  'Amalie Welke',
  'Ewald Rapp',
  'Del Robins',
  'Susanna Roundy',
  'Mariea Kalbert',
  'Georg Rapp',
  'John Stewart',
  'Regina Stewart'
];

var m = new Array(people.length);
var i;
for (i = 0; i < 17; i++) {
  m[i] = new Array(people.length);
}

m[0][0] = 0;
m[0][7] = 3;
m[1][1] = 1;
m[1][5] = 2;
m[1][12] = 3;
m[2][2] = 1;
m[2][10] = 2;
m[2][16] = 3;
m[3][3] = 0;
m[3][14] = 2;
m[3][13] = 3;
m[4][4] = 0;
m[5][5] = 0;
m[6][6] = 0;
m[6][10] = 2;
m[6][16] = 3;
m[7][7] = 1;
m[8][8] = 1;
m[9][9] = 1;
m[10][10] = 0;
m[10][3] = 2;
m[10][9] = 3;
m[11][11] = 1;
m[12][12] = 1;
m[13][13] = 1;
m[14][14] = 0;
m[15][15] = 0;
m[15][4] = 2;
m[15][9] = 3;
m[16][16] = 1;
m[16][15] = 2;
m[16][1] = 3;

console.table(m);

function generation(x) {
  if (x < 0) return undefined;
  if (x == 0 || x == 1) return 0;

  var i = 0;
  while (true) {
    if ((x / Math.pow(2, i)) >> 0 == 0) {
      break;
    }
    i++;
  }
  return i - 1;
}

function avos(a, b) {
  if (typeof(a) == 'number' && typeof(b) == 'number') {
    var generationNumber = generation(b);
    return (b & (Math.pow(2, generationNumber) - 1)) | (a << generationNumber);
  }
  return 0;
}

function expand(m) {
  var result = new Array(m.length);
  for (var i = 0; i < m.length; i++) {
    result[i] = new Array(m[i].length);
    for (var j = 0; j < m[0].length; j++) {
      if (typeof(m[i][j]) == 'undefined') {
        var sum = 0;
        for (var k = 0; k < m.length; k++) {
          sum |= avos(m[k][j], m[i][k]);
        }
        if (sum != 0) {
          result[i][j] = sum;
        }
      }
      else {
        result[i][j] = m[i][j];
      }
    }
  }
  return result;
}

m = expand(m);
console.table(m);
m = expand(m);
console.table(m);
m = expand(m);
console.table(m);


