var slideWrapper = $(".main-slider"),
  iframes = slideWrapper.find(".embed-player"),
  lazyImages = slideWrapper.find(".slide-image"),
  lazyCounter = 0;

// POST commands to YouTube or Vimeo API
function postMessageToPlayer(player, command) {
  if (player == null || command == null) return;
  player.contentWindow.postMessage(JSON.stringify(command), "*");
}

// When the slide is changing
function playPauseVideo(slick, control) {
  var currentSlide, slideType, startTime, player, video;

  currentSlide = slick.find(".slick-current");
  slideType = currentSlide.attr("class").split(" ")[1];
  player = currentSlide.find("iframe").get(0);
  startTime = currentSlide.data("video-start");

  if (slideType === "vimeo") {
    switch (control) {
      case "play":
        if (
          startTime != null &&
          startTime > 0 &&
          !currentSlide.hasClass("started")
        ) {
          currentSlide.addClass("started");
          postMessageToPlayer(player, {
            method: "setCurrentTime",
            value: startTime
          });
        }
        postMessageToPlayer(player, {
          method: "play",
          value: 1
        });
        break;
      case "pause":
        postMessageToPlayer(player, {
          method: "pause",
          value: 1
        });
        break;
    }
  } else if (slideType === "youtube") {
    switch (control) {
      case "play":
        postMessageToPlayer(player, {
          event: "command",
          func: "mute"
        });
        postMessageToPlayer(player, {
          event: "command",
          func: "playVideo"
        });
        break;
      case "pause":
        postMessageToPlayer(player, {
          event: "command",
          func: "pauseVideo"
        });
        break;
    }
  } else if (slideType === "video") {
    video = currentSlide.children("video").get(0);
    if (video != null) {
      if (control === "play") {
        video.play();
      } else {
        video.pause();
      }
    }
  }
}

// Resize player
function resizePlayer(iframes, ratio) {
  if (!iframes[0]) return;
  var win = $(".main-slider"),
    width = win.width(),
    playerWidth,
    height = win.height(),
    playerHeight,
    ratio = ratio || 16 / 9;

  iframes.each(function () {
    var current = $(this);
    if (width / ratio < height) {
      playerWidth = Math.ceil(height * ratio);
      current
        .width(playerWidth)
        .height(height)
        .css({
          left: (width - playerWidth) / 2,
          top: 0
        });
    } else {
      playerHeight = Math.ceil(width / ratio);
      current
        .width(width)
        .height(playerHeight)
        .css({
          left: 0,
          top: (height - playerHeight) / 2
        });
    }
  });
}

// DOM Ready
$(function () {
  // Initialize
  slideWrapper.on("init", function (slick) {
    slick = $(slick.currentTarget);
    setTimeout(function () {
      playPauseVideo(slick, "play");
    }, 1000);
    resizePlayer(iframes, 16 / 9);
  });
  slideWrapper.on("beforeChange", function (event, slick) {
    slick = $(slick.$slider);
    playPauseVideo(slick, "pause");
  });
  slideWrapper.on("afterChange", function (event, slick) {
    slick = $(slick.$slider);
    playPauseVideo(slick, "play");
  });
  slideWrapper.on("lazyLoaded", function (event, slick, image, imageSource) {
    lazyCounter++;
    if (lazyCounter === lazyImages.length) {
      lazyImages.addClass("show");
      // slideWrapper.slick("slickPlay");
    }
  });

  //start the slider
  slideWrapper.slick({
    // fade:true,
    autoplaySpeed: 4000,
    lazyLoad: "progressive",
    speed: 600,
    arrows: false,
    dots: true,
    cssEase: "cubic-bezier(0.87, 0.03, 0.41, 0.9)"
  });
});

// Resize event
$(window).on("resize.slickVideoPlayer", function () {
  resizePlayer(iframes, 16 / 9);
});


$(document).ready(function () {
  initFileUploader("#zdrop");
  function initFileUploader(target) {
    var previewNode = document.querySelector("#zdrop-template");
    previewNode.id = "";
    var previewTemplate = previewNode.parentNode.innerHTML;
    previewNode.parentNode.removeChild(previewNode);

    var zdrop = new Dropzone(target, {
      url: "upload.php",
      maxFiles: 1,
      maxFilesize: 30,
      previewTemplate: previewTemplate,
      previewsContainer: "#previews",
      clickable: "#upload-label"
    });

    zdrop.on("addedfile", function (file) {
      $(".preview-container").css("visibility", "visible");
    });

    zdrop.on("totaluploadprogress", function (progress) {
      var progr = document.querySelector(".progress .determinate");
      if (progr === undefined || progr === null) return;

      progr.style.width = progress + "%";
    });

    zdrop.on("dragenter", function () {
      $(".fileuploader").addClass("active");
    });

    zdrop.on("dragleave", function () {
      $(".fileuploader").removeClass("active");
    });

    zdrop.on("drop", function () {
      $(".fileuploader").removeClass("active");
    });
  }
});


// 비디오 업로드 Dropzone 설정
function initVideoUploader() {
  var myDropzone = new Dropzone("#video-upload", {
      url: "upload.php",
      maxFiles: 1,
      acceptedFiles: "video/*",
      previewTemplate: document.querySelector('#zdrop-template').innerHTML
  });

  myDropzone.on("addedfile", function(file) {
      console.log("Added file:", file.name);
  });

  myDropzone.on("uploadprogress", function(file, progress) {
      console.log("File progress:", progress);
  });

  myDropzone.on("success", function(file) {
      console.log("File uploaded:", file.name);
  });

  myDropzone.on("error", function(file, response) {
      console.log("Error during upload:", response);
  });

  myDropzone.on("removedfile", function(file) {
      console.log("File removed:", file.name);
  });
}

$(document).ready(function() {
  initVideoUploader();
});


