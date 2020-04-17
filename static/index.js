// document.getElementById("subbutton").addEventListener("click", () => {
//     let xhttp = new XMLHttpRequest();
//     xhttp.onreadystatechange = () => {
//         if (xhttp.readyState == 4 && xhttp.status == 200) {
//             if (xhttp.responseText == 0) {
//                 // Okay, get angry.
//             } else {
//                 // Our code knows we're good now; let's help ourselves out by reloading the page with all our info.
//                 window.location.reload(true)
//             }
//         }
//     }
//     xhttp.open("POST", "myInfo", true)
//     xhttp.setRequestHeader("content-type", "application/x-www-form-urlencoded")
//     xhttp.send(`accountName=${document.getElementById("feelinit").value}&password=${document.getElementById("eventsitup").value}`)
// })