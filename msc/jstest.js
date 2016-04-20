var client = require('ar-drone').createClient();
const addon = require('./build/Release/addon');
var fs = require('fs');

var pngStream = client.getPngStream();

var lastPng;
pngStream
  .on('error', console.log)
  .on('data', function(pngBuffer) {
    lastPng += pngBuffer;
  });

console.log(‘It is a buffer:’+isBuffer(lastPng));

// fs.writeFile('logo.png', lastPng, 'binary', function(err){
            // if (err) throw err
            // console.log('File saved.')
        // })

//var vdata;
// var buf = new Buffer ([0x62,0x75,0x66,0x66,0x65,0x72]);
//var buf = new Buffer ([1,2,3,4,5]);
//imTest(buf);

// video.on('data', imTest.imTest);
// console.log('This shoul');
console.log(addon.isbuf(lastPng));