module.exports = {

  generation: function (x) {
    if (x < 0) return undefined;
    //noinspection StatementWithEmptyBodyJS
    for(var i = 0; x > 0; x = x >> 1, i++);
    return i > 0 ? i - 1 : i;
  },

  avos: function(a, b) {
    if (typeof(a) == 'number' && typeof(b) == 'number') {
      var generationNumber = this.generation(b);
      if (a == 0 || a == 1) {
        return (generationNumber == 0 && a != b) ? undefined : b;
      }
      return (b & (Math.pow(2, generationNumber) - 1)) | (a << generationNumber);
    }
    return undefined;
  },

  expand: function(m) {
    if (m.length != m[0].length) return undefined;
    var result = new Array(m.length);
    for (var i = 0; i < m.length; i++) {
      result[i] = new Array(m[i].length);
      for (var j = 0; j < m[0].length; j++) {
        // only calculate a new result cell if the cell in the input matrix
        // has not yet been defined
        if (typeof(m[i][j]) == 'undefined') {
          var sum = 0;
          for (var k = 0; k < m.length; k++) {
            sum |= this.avos(m[i][k], m[k][j]);
          }
          if (sum != 0) {
            result[i][j] = sum;
          }
        }
        else {
          // if it has been defined, just copy it over
          result[i][j] = m[i][j];
        }
      }
    }
    return result;
  }

};
