// global
var url = '';
var data = [];

// image click
$(".img").click(function() {

    // empty/hide results
    $("#results").empty();
    $("#results-table").hide();
    $("#error").hide();

    // remove active class
    $(".img").removeClass("active")

    // add active class to clicked picture
    $(this).addClass("active")

    // grab image url
    var image = $(this).attr("src")
    console.log(image)
    //alert(image)
    alert("start searching...")
    // show searching text
    $("#searching").empty();
    $("#searching").append('Searching...');
    $("#searching").show();
    console.log("searching...")

    // ajax request
    $.ajax({
      type: "POST",
      url: "/search",
      data : { img : image },
      // handle success
      success: function(result) {
        console.log(result.results);
        var data = result.results;
        var classification = result.classification;
        // show table
        $("#searching").empty();
        $("#searching").append(classification);
        $("#results-table").show();
        // loop through results, append to dom
        for (i = 0; i < data.length; i++) {
          $("#results").append('<tr><th><a href="'+data[i]["image"]+'"><img src="'+data[i]["image"]+
            '" class="result-img"></a></th><th>'+data[i]['score']+'</th></tr>')
        };
      },
      // handle error
      error: function(error) {
        console.log(error);
        // append to dom
        $("#error").append()
      }
    });
});