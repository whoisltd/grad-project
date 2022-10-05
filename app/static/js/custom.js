/*  ==========================================
    SHOW UPLOADED IMAGE
* ========================================== */
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function () {
        readURL($('#upload'));
    });
});

/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
// var input = ;
var infoArea = $('#upload-label');


function showFileName(event) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}

$('#upload').bind('change',showFileName);
$(function() {
    $('#submit-button').click(function() {
        // var form_data = new FormData($('#upload-img'));
        $.ajax({
            type: 'POST',
            url: '/api/v1/ocr',
            data:  JSON.stringify({img: $("#imageResult")[0].src.split(',')[1]}),
            contentType: 'app/json',
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
                console.log(data);
                $('#id').val(data['data']['data']['id']);
                $('#name').val(data['data']['data']['name']);
                $('#birthday').val(data['data']['data']['birthday']);
                $('#nation').val(data['data']['data']['nation']);
                $('#sex').val(data['data']['data']['sex']);
                $('#exp').val(data['data']['data']['exp']);
                $('#hometown').val(data['data']['data']['hometown']);
                $('#addr').val(data['data']['data']['addr']);
            },
        });
    });
});