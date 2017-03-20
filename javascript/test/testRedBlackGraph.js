var assert = require('assert');
require('console.table');
var redBlackGraph = require('../src/redBlackGraph');

describe('redBlackGraph', function() {
  describe('#generation()', function () {
    it('should return undefined if position id is negative', function() {
      assert.equal(undefined, redBlackGraph.generation(-1));
    });
    it('should return 0 if position id is either 0 or 1', function() {
      assert.equal(0, redBlackGraph.generation(0));
      assert.equal(0, redBlackGraph.generation(1));
    });
    it('should return generation number for position id greater than 1', function() {
      assert.equal(1, redBlackGraph.generation(2));
      assert.equal(1, redBlackGraph.generation(3));
      assert.equal(2, redBlackGraph.generation(4));
      assert.equal(2, redBlackGraph.generation(7));
      assert.equal(3, redBlackGraph.generation(8));
      assert.equal(3, redBlackGraph.generation(15));
      assert.equal(4, redBlackGraph.generation(16));
    });
  });
  describe('#avos()', function() {
    it('validate self (male)', function() {
      assert.equal(redBlackGraph.avos(0, 0), 0);
      assert.equal(redBlackGraph.avos(0, 1), undefined);
    });
    it('validate self (female)', function() {
      assert.equal(redBlackGraph.avos(1, 0), undefined);
      assert.equal(redBlackGraph.avos(1, 1), 1);
    });
    it('validate self (male) parents', function() {
      assert.equal(redBlackGraph.avos(0, 2), 2);
      assert.equal(redBlackGraph.avos(0, 3), 3);
    });
    it('validate self (female) parents', function() {
      assert.equal(redBlackGraph.avos(1, 2), 2);
      assert.equal(redBlackGraph.avos(1, 3), 3);
    });
    it('validate father\'s parents', function() {
      assert.equal(redBlackGraph.avos(2, 2), 4);
      assert.equal(redBlackGraph.avos(2, 3), 5);
    });
    it('validate mother\'s parents', function() {
      assert.equal(redBlackGraph.avos(3, 2), 6);
      assert.equal(redBlackGraph.avos(3, 3), 7);
    });
    it('validate father\'s grandparents', function() {
      assert.equal(redBlackGraph.avos(2, 4), 8);
      assert.equal(redBlackGraph.avos(2, 5), 9);
      assert.equal(redBlackGraph.avos(2, 6), 10);
      assert.equal(redBlackGraph.avos(2, 7), 11);

      assert.equal(redBlackGraph.avos(2, 8), 16);
      assert.equal(redBlackGraph.avos(2, 9), 17);
      assert.equal(redBlackGraph.avos(2, 10), 18);
      assert.equal(redBlackGraph.avos(2, 11), 19);
      assert.equal(redBlackGraph.avos(2, 12), 20);
      assert.equal(redBlackGraph.avos(2, 13), 21);
      assert.equal(redBlackGraph.avos(2, 14), 22);
      assert.equal(redBlackGraph.avos(2, 15), 23);
    });
    it('validate mother\'s grandparents', function() {
      assert.equal(redBlackGraph.avos(3, 4), 12);
      assert.equal(redBlackGraph.avos(3, 5), 13);
      assert.equal(redBlackGraph.avos(3, 6), 14);
      assert.equal(redBlackGraph.avos(3, 7), 15);

      assert.equal(redBlackGraph.avos(3, 8), 24);
      assert.equal(redBlackGraph.avos(3, 9), 25);
      assert.equal(redBlackGraph.avos(3, 10), 26);
      assert.equal(redBlackGraph.avos(3, 11), 27);
      assert.equal(redBlackGraph.avos(3, 12), 28);
      assert.equal(redBlackGraph.avos(3, 13), 29);
      assert.equal(redBlackGraph.avos(3, 14), 30);
      assert.equal(redBlackGraph.avos(3, 15), 31);
    });
    it('validate paternal grandfather\'s parents', function() {
      assert.equal(redBlackGraph.avos(4, 2), 8);
      assert.equal(redBlackGraph.avos(4, 3), 9);
    });
    it('validate paternal grandmother\'s parents', function() {
      assert.equal(redBlackGraph.avos(5, 2), 10);
      assert.equal(redBlackGraph.avos(5, 3), 11);
    });
    it('validate maternal grandfather\'s parents', function() {
      assert.equal(redBlackGraph.avos(6, 2), 12);
      assert.equal(redBlackGraph.avos(6, 3), 13);
    });
    it('validate maternal grandmother\'s parents', function() {
      assert.equal(redBlackGraph.avos(7, 2), 14);
      assert.equal(redBlackGraph.avos(7, 3), 15);
    });
  });
  describe('#expand() - simple', function() {
    it('"squaring" the initial matrix shoudl result in discovering grandparent relationships', function() {
      var people = [
        'Daniel Rapp',            // 0
        'Ewald Rapp',             // 1
        'Regina Stewart',         // 2
        'John Stewart',           // 3
        'Linette Inez Williams',  // 4
        'Michael Joseph Rapp',    // 5
        'Amalie Welke'            // 6
      ];

      var m = new Array(people.length);
      var i;
      for (i = 0; i < people.length; i++) {
        m[i] = new Array(people.length);
      }

      m[0][0] = 0;
      m[0][1] = 2;
      m[0][2] = 3;
      m[1][1] = 0;
      m[1][5] = 2;
      m[1][6] = 3;
      m[2][2] = 1;
      m[2][3] = 2;
      m[2][4] = 3;
      m[3][3] = 0;
      m[4][4] = 1;
      m[5][5] = 0;
      m[6][6] = 1;

      var expansion1 = redBlackGraph.expand(m);
      // only modification should be first row, verify all other rows intact
      for (i = 1; i < people.length; i++) {
        assert.deepEqual(m[i], expansion1[i]);
      }
      assert.equal(0, expansion1[0][0], 'calculation incorrect for ' + people[0]);
      assert.equal(2, expansion1[0][1], 'calculation incorrect for ' + people[2]);
      assert.equal(3, expansion1[0][2], 'calculation incorrect for ' + people[3]);
      assert.equal(6, expansion1[0][3], 'calculation incorrect for ' + people[3]);
      assert.equal(7, expansion1[0][4], 'calculation incorrect for ' + people[4]);
      assert.equal(4, expansion1[0][5], 'calculation incorrect for ' + people[5]);
      assert.equal(5, expansion1[0][6], 'calculation incorrect for ' + people[6]);
    })
  });
  describe('#expand() - moderately complex', function() {
    it('"squaring" the initial matrix shoudl result in discovering grandparent relationships', function() {
      var people = [
        'Jiovan Melendez',        // 0
        'Linette Inez Williams',  // 1
        'Marianna Rapp',          // 2
        'Michael Joseph Rapp',    // 3
        'Samuel Stewart',         // 4
        'John Davis Williams',    // 5
        'Daniel Rapp',            // 6
        'Barbara Valcarcel',      // 7
        'Emeline Chidester',      // 8
        'Amalie Welke',           // 9
        'Ewald Rapp',             // 10
        'Del Robins',             // 11
        'Susanna Roundy',         // 12
        'Mariea Kalbert',         // 13
        'Georg Rapp',             // 14
        'John Stewart',           // 15
        'Regina Stewart',         // 16
        'Iris Gonzalez'           // 17
      ];

      var m = new Array(people.length);
      var i;
      for (i = 0; i < people.length; i++) {
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
      m[7][17] = 3;
      m[8][8] = 1;
      m[9][9] = 1;
      m[10][10] = 0;
      m[10][3] = 2;
      m[10][9] = 3;
      m[11][11] = 0;
      m[12][12] = 1;
      m[13][13] = 1;
      m[14][14] = 0;
      m[15][15] = 0;
      m[15][4] = 2;
      m[15][9] = 3;
      m[16][16] = 1;
      m[16][15] = 2;
      m[16][1] = 3;
      m[17][17] = 1;

      var expansion1 = redBlackGraph.expand(m);
      assert.notDeepEqual(m, expansion1);
      var expansion2 = redBlackGraph.expand(expansion1);
      assert.notDeepEqual(expansion1, expansion2);
      var expansion3 = redBlackGraph.expand(expansion2);
      assert.deepEqual(expansion2, expansion3);
      console.table(expansion3);

      // now validate a few of the computed relationships
      assert.equal(expansion3[0][17], 7, 'Iris should be Jiovan\'s maternal grandmother');
      assert.equal(expansion3[2][1], 7, 'Linette should be Mariannaa\'s maternal grandmother');
      assert.equal(expansion3[2][3], 4, 'Michael should be Mariannaa\'s paternal grandfather');
      assert.equal(expansion3[2][4], 12, 'Samuel should be Mariannaa\'s MFF');
      assert.equal(expansion3[2][5], 14, 'John should be Mariannaa\'s MMF');
      assert.equal(expansion3[2][9], 5, 'Amalie should be Mariannaa\'s patlernal grandmother');
      assert.equal(expansion3[2][12], 15, 'Susanna should be Mariannaa\'s MMM');
      assert.equal(expansion3[2][13], 9, 'Mariea should be Mariannaa\'s FFM');
      assert.equal(expansion3[2][14], 8, 'George should be Mariannaa\'s FFF');
      assert.equal(expansion3[2][15], 6, 'John should be Mariannaa\'s maternal grandfather');
    })
  });
});