var client = require('ar-drone').createClient();
const myaddon = require('../build/Release/addon');
var addon = myaddon();
var fs = require('fs');


var pngStream = client.getPngStream();
client.config('video:video_channel', 0);

var lastPng;
pngStream
  .on('error', console.log)
// .on('data', function (pngbuffer) {
//    lastPng = pngbuffer;
//    //console.log("it is a buffer:" + buffer.isbuffer(lastpng));
//    //console.log(addon(lastPng).spin);
//    //console.log(Date.toString());
//    var ctrlData = addon(lastPng);
//    console.log(0);
//    console.log(ctrlData.spin);
//})
//.once('data', lastPng);

var timerId;
const interval = 100;
setTimeout(function () { } , 1000);

process.on('SIGINT', function () {
    console.log('Got SIGINT. Landing, press Control-C again to force exit.');
	//clearInterval(timerId);
	client.land();
    setTimeout(function () {
        console.log("Landing done.");
		console.log("battery level:",client.battery());
        //client.land();
        process.exit(0);
    }, 1000);
});


client
    .takeoff();

client
    .after(1000,function() {
    if (1) {
        //timerId = setInterval(function (lastPng) {
            
            pngStream
        .on('data', function (pngBuffer) {
				console.time("C time");
                lastPng = pngBuffer;
                //console.log("It is a buffer:" + Buffer.isBuffer(lastPng));
                //console.log(addon.myctrl(lastPng));
                var ctrlData = addon(lastPng);
                //console.log(ctrlData);
                if (ctrlData.spin > 0) {
                    client.clockwise(ctrlData.spin);
                }
                else if (ctrlData.spin < 0) {
                    client.counterClockwise(-ctrlData.spin);
                }
                else {
                    client.stop();
                }
				console.timeEnd("C time");
            //process.on('SIGINT', function () {
            //    console.log('Got SIGINT. Landing, press Control-C again to force exit.');
            //    setTimeout(function () {
            //        console.log("Landing drone.");
            //        client.land(function () {
            //            process.exit(0);
            //        });
            //        clearInterval(timerId);
            //    }, 1000);
            //});
            })
        //}, interval);
    };
});






//fs.writeFile('./logo.png', lastPng, 'binary', function(err){
//            if (err) throw err
//            console.log('File saved.')
//        })

//var vdata;
// var buf = new Buffer ([0x62,0x75,0x66,0x66,0x65,0x72]);
//var buf = new Buffer ([1,2,3,4,5]);
//imTest(buf);

// video.on('data', imTest.imTest);
// console.log('This shoul');

//console.log(addon.isbuf(lastPng));

