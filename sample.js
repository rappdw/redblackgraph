require('console.table');

var m = new Array(7);
var n = new Array(7);
var i;
for (i = 0; i < 7; i++) {
  m[i] = new Array(7);
  n[i] = new Array(7);
  m[i][i] = i & 1;
}

m[0][1] = 2;
m[0][2] = 3;
m[1][3] = 2;
m[1][4] = 3;
m[2][5] = 2;
m[2][6] = 3;

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
  var result = [];
  for (var i = 0; i < m.length; i++) {
    result[i] = [];
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


